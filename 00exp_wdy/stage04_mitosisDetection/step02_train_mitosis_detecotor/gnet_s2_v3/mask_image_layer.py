import caffe

import numpy as np
import cv2
from numpy import random
import random as pyrandom

def vis_square(data):
    """Take an array of shape (n, height, width) or (n, height, width, 3)
       and visualize each (height, width) thing in a grid of size approx. sqrt(n) by sqrt(n)"""

    # normalize data for display
    #data = (data - data.min()) / (data.max() - data.min())

    # force the number of filters to be square
    n = int(np.ceil(np.sqrt(data.shape[0])))
    padding = (((0, n ** 2 - data.shape[0]),
               (0, 1), (0, 1))                 # add some space between filters
               + ((0, 0),) * (data.ndim - 3))  # don't pad the last dimension (if there is one)
    data = np.pad(data, padding, mode='constant', constant_values=1)  # pad with ones (white)

    # tile the filters into an image
    data = data.reshape((n, n) + data.shape[1:]).transpose((0, 2, 1, 3) + tuple(range(4, data.ndim + 1)))
    data = data.reshape((n * data.shape[1], n * data.shape[3]) + data.shape[4:])
    return data

class random_patches_from_images_withlabel(caffe.Layer):
    def setup(self, bottom, top):
        # config
        params = eval(self.param_str)
        self.root_folder = params['root_folder']
        self.image_list = params['image_list']
        self.size = params['size']
        self.batch = params['batch']
        self.mean = np.array(params['mean'])
        self.random = params.get('randomize', True)
        self.seed = params.get('seed', None)
        self.scale = params.get('scale', 0.0)
        self.colorn = params.get('colorn', 0)
        
        # two tops: data and label
        if len(top) != 2:
            raise Exception("Need to define one top: data and label")
        # data layers have no bottoms
        if len(bottom) != 0:
            raise Exception("Do not define a bottom.")

        # load indices for images and labels
        self.images = ['%s/%s'%(self.root_folder, l.strip().split()[0]) for l in open(self.image_list)]
        self.labels = ['%s/%s'%(self.root_folder, l.strip().split()[1]) for l in open(self.image_list)]
        self.idx = 0

        if self.random:
            random.seed(self.seed)
            pyrandom.seed(self.seed)
        if self.size % 2 == 0:
            self.lpsize = self.size/2
            self.rpsize = self.size/2
        else:
            self.lpsize = (self.size - 1)/2
            self.rpsize = (self.size - 1)/2 + 1
      
    def reshape(self, bottom, top):
        self.data, self.label = self.load_data(self.idx)
        top[0].reshape(*self.data.shape)
        top[1].reshape(*self.label.shape)

    def forward(self, bottom, top):
        top[0].data[...] = self.data
        top[1].data[...] = self.label
        if self.random:
            self.idx = random.randint(len(self.images))
        else:
            self.idx += 1
            if self.idx == len(self.indices):
                self.idx = 0

    def backward(self, top, propagate_down, bottom):
        pass

    def crop_patch(self, img, y, x, ls, rs):
        h, w, c  = img.shape
        # scale
        scale = 1.0
        if self.scale > 0:
            if pyrandom.random() > 0.5:
                scale += self.scale
            else:
                scale -= self.scale
        ls_n = np.floor(ls * scale)
        rs_n = np.floor(rs * scale)
        
        p = None
        if y - ls_n < 0 or x - ls_n < 0 or y + rs_n >= h or x + rs_n >= w:
            return None  
        else:
            p = img[y-ls_n:y+rs_n, x-ls_n:x+rs_n, :]
            #print p.shape
            p = cv2.resize(p, (self.size, self.size))
        # rotation
        p = np.rot90(p, random.randint(4))                              
        # flip
        if pyrandom.random() > 0.5:
            p = np.fliplr(p)
        if pyrandom.random() > 0.5:
            p = np.flipud(p)
        # color noise and remove mean
        if self.colorn > 0:
            for ii in range(c):
                p_ii = p[:,:,ii] 
                if pyrandom.random() > 0.5:
                    p_ii += random.randint(self.colorn)
                else:
                    p_ii -= random.randint(self.colorn)
                p_ii[p_ii < 0] = 0
                p_ii[p_ii > 255] = 255
                p[:,:,ii] = p_ii - self.mean[ii]
        return p.transpose((2,0,1)) # c h w
        
    def load_data(self, idx):
        print "Load Image:%s"%(self.images[idx])
        img = cv2.imread(self.images[idx])
        msk = cv2.imread(self.labels[idx], cv2.CV_LOAD_IMAGE_GRAYSCALE)

        in_ = np.array(img, dtype=np.float32)
        #in_ = in_.transpose((2,0,1)) # channel x height x width
        #in_ -= self.mean # remove mean
        h, w, c =in_.shape

        ys, xs = np.where(msk==2)        
        pos_locs = [(y, x) for y, x in zip(ys, xs)]
        ys, xs = np.where(msk==1)
        neg_locs = [(y, x) for y, x in zip(ys, xs)]
        
        #pos_num = np.min([self.batch/2, len(pos_locs)])
        #neg_num = self.batch - pos_num
        
        pyrandom.shuffle(pos_locs)
        pyrandom.shuffle(neg_locs)
        imgs = np.zeros((self.batch, c, self.size, self.size), dtype=np.float32)
        lals = []
        p_ct = 0
        for y, x in pos_locs:
            p = in_[:, y-self.lpsize : y+self.rpsize, x-self.lpsize : x+self.rpsize]
            p = self.crop_patch(in_, y, x, self.lpsize, self.rpsize)
            if p is not None:
                imgs[p_ct, :, :, :] = p
                lals.append(1)
                p_ct = p_ct + 1
                if p_ct == self.batch / 2:
                    break
        n_ct = 0
        for y, x in neg_locs:
            p = self.crop_patch(in_, y, x, self.lpsize, self.rpsize)
            if p is not None:
                imgs[p_ct+n_ct, :, :, :] = p
                lals.append(0)
                n_ct = n_ct + 1
                if n_ct == self.batch - p_ct:
                    break
        
        lals = np.asarray(lals).reshape((-1, 1, 1, 1))

        DEBUG=1
        if DEBUG:
            vimg = vis_square(imgs.transpose(0, 2, 3, 1))
            vimg += self.mean # remove mean
            #vimg = vimg[:,:,::-1] # BGR - > RGB
            #vim = vimg.astype(np.uint8)            
            cv2.imwrite('images/a_%010d.png'%idx, vimg)
        return imgs, lals
