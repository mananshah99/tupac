import os, sys, csv
import skimage.io as skio
import skimage as ski

from skimage.measure import label
from scipy.signal import convolve
from skimage.morphology import disk
from skimage.morphology import dilation
from skimage.feature import peak_local_max
import numpy as np
from multiprocessing import Pool
from scipy.spatial import distance
import skimage.filters as skif
from skimage.measure import regionprops

ori_root='/home/dywang/Proliferation/data/mitoses/mitoses_image_data'

#input_folders = ['fnet.M01/aug_top12_0050K/mitoses_image_data_cn3']
#t=0.50 TPs=373 FPs=1290 FNs=160
#0.22429344558 0.699812382739 0.33970856102

#input_folders = ['fnet.M01/aug_top12_0650K/mitoses_image_data_cn3']
#t=0.50 TPs=325 FPs=675 FNs=208
#0.325 0.609756097561 0.424005218526
#t=0.80 s=20 TPs=191 FPs=136 FNs=342
#0.584097859327 0.358348968105 0.444186046512

#input_folders = ['fnet.M01/aug_top12_1000K/mitoses_image_data_cn3']
#t=0.50 TPs=294 FPs=683 FNs=239
#0.300921187308 0.551594746717 0.38940397351

#input_folders = ['fnet.M16/aug_top12_1000K/mitoses_image_data_cn3']
#t=0.5 TPs=261 FPs=233 FNs=272
#0.528340080972 0.489681050657 0.508276533593

#input_folders = ['fnet.M01/aug_top12_0050K/mitoses_image_data_cn3', 'fnet.M16/aug_top12_1000K/mitoses_image_data_cn3']
#t=0.50 TPs=234 FPs=111 FNs=299
#0.678260869565 0.439024390244 0.533029612756

#input_folders = ['fnet.M01/aug_top12_0650K/mitoses_image_data_cn3', 'fnet.M01/aug_top12_1000K/mitoses_image_data_cn3', 'fnet.M16/aug_top12_1000K/mitoses_image_data_cn3']
#t=0.50 TPs=249 FPs=168 FNs=284
#0.597122302158 0.467166979362 0.524210526316

################################################################################################################################i
#input_folders = ['mnet_v2_aug/stage02/top12_model01_conf_0250K/mitoses_image_data_cn3']
#t=0.50 s=20 TPs=367 FPs=4615 FNs=166
#0.0736651947009 0.688555347092 0.13309156845
#t=0.80 s=20 TPs=182 FPs=437 FNs=351
#0.294022617124 0.341463414634 0.315972222222

#input_folders = ['mnet_v2_aug/stage02/top12_model01_conf_0500K/mitoses_image_data_cn3']
#t=0.50 s=20 TPs=284 FPs=954 FNs=249
#0.229402261712 0.532833020638 0.320722755505

#input_folders = ['mnet_v2_aug/stage02/top12_model01_conf_1000K/mitoses_image_data_cn3']
#t=0.50 s=20 TPs=402 FPs=7372 FNs=131
#0.051710830975 0.754221388368 0.0967858432647
#t=0.80 s=20 TPs=220 FPs=1111 FNs=313
#0.165289256198 0.412757973734 0.236051502146

#input_folders = ['mnet_v2_aug/stage02/top12_model01_conf_1500K/mitoses_image_data_cn3']
#t=0.50 s=20 TPs=307 FPs=1356 FNs=226
#0.184606133494 0.575984990619 0.279599271403

#input_folders = ['mnet_v2_aug/stage02/top12_model01_conf_2000K/mitoses_image_data_cn3']
#t=0.50 s=20 TPs=321 FPs=2482 FNs=212
#0.114520156975 0.602251407129 0.192446043165

#input_folders = ['mnet_v2_aug/stage02/top12_model01_conf_0250K/mitoses_image_data_cn3', 'mnet_v2_aug/stage02/top12_model01_conf_0500K/mitoses_image_data_cn3', 'mnet_v2_aug/stage02/top12_model01_conf_1500K/mitoses_image_data_cn3', 'mnet_v2_aug/stage02/top12_model01_conf_2000K/mitoses_image_data_cn3']
#t=0.80 s=20 TPs=105 FPs=69 FNs=428
#0.603448275862 0.196998123827 0.29702970297

################################################################################################################################i
#input_folders = ['fnet.M01/aug_top12_0050K/mitoses_image_data_cn3', 'mnet_v2_aug/stage02/top12_model01_conf_1000K/mitoses_image_data_cn3']
#t=0.80 s=20 TPs=201 FPs=161 FNs=332
#0.555248618785 0.377110694184 0.449162011173

################################################################################################################################i
#input_folders = ['mnet_v2_aug.M16/stage02/top12_model03_conf_1000K/heatmap/mitoses_image_data_cn3']
#t=0.50 s=20 TPs=390 FPs=2836 FNs=143
#0.120892746435 0.731707317073 0.207501995211

################################################################################################################################i
#input_folders = ['mnet_v2_aug/stage02/top12_model02/mitoses_image_data_cn3']
#t=0.50 TPs=308 FPs=1011 FNs=225
#0.233510235027 0.577861163227 0.332613390929

#input_folders = ['mnet_v2_aug/stage02/top12_model03/mitoses_image_data_cn3']
#t=0.50 s=20 TPs=365 FPs=3446 FNs=168
#0.0957753870375 0.684803001876 0.168047882136

EXT = '_Normalized.tif.png'

#input_folders = ['mitko/model01/mitoses_image_data_cn1'] # 0.52 (t=0.8)
#EXT = '.tif_nc.png.png'
#t=0.50 s=20 TPs=444 FPs=1506 FNs=89
#0.227692307692 0.833020637899 0.357631896899
#t=0.50 s=50 TPs=424 FPs=1109 FNs=109
#0.276581865623 0.795497185741 0.410454985479
#t=0.50 s=100 TPs=396 FPs=680 FNs=137
#0.368029739777 0.74296435272 0.492231199503
#t=0.8 s=20 TPs=310 FPs=259 FNs=223
#0.544815465729 0.581613508443 0.562613430127
#t=0.80 s=50 TPs=303 FPs=231 FNs=230
#0.567415730337 0.568480300188 0.567947516401
#t=0.80 s=100 TPs=226 FPs=112 FNs=307
#0.668639053254 0.424015009381 0.518943742824

ground_folder = 'ground'
#lst = 'tr.lst'
lst = 'te.lst'

#img_paths = ['%s/%s_Normalized.tif.png'%(input_folder, l.strip()) for l in open(lst)]
img_paths = []

for l in open(lst):
    img_paths_i = ['%s/%s%s'%(input_folder, l.strip(), EXT) for input_folder in input_folders]
    img_paths_i.append('%s/%s_mask_nuclei.png'%(ori_root, l.strip()))
    img_paths.append(img_paths_i)
grd_paths = ['%s/%s.csv'%(ground_folder, l.strip()) for l in open(lst)]

tv = [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]

def get_a_file(file_path, mode):
    file_folder = os.path.dirname(file_path)
    if not os.path.exists(file_folder):
        print("Making new folder: ", file_folder)
        os.makedirs(file_folder)
    if mode is not None:
        f = open(file_path, mode)
        return f
    else: # we only need the folder
        return None

def load_pts(grd_path):
    pts = []
    if  os.path.exists(grd_path):
        with open(grd_path, 'r') as csvfile:
            spamreader = csv.reader(csvfile)
            for line in spamreader:
                row, col = line[0], line[1]
                pts.append((int(row), int(col)))
    return pts

def gen_filter(radius):
    kernel = np.zeros((2*radius+1, 2*radius+1))
    y,x = np.ogrid[-radius:radius+1, -radius:radius+1]
    mask = x**2 + y**2 <= radius**2
    kernel[mask] = 1
    return kernel

def get_position(param):
    img_ID, img_path, tv, sz, radius, mindis, result_folder = param
    print "Processing:%s (tv=%.2f radius=%.2f mindis=%.2f)"%(img_path, tv, radius, mindis)

    # load images
    for img_id, img_path_i in enumerate(img_path[:-1]):
        if img_id == 0:
            hp = skio.imread(img_path_i) / 255.0
        else:
            hp+= skio.imread(img_path_i) / 255.0
    hp_bg_msk = skio.imread(img_path[-1])  == 0
    #hp[hp_bg_msk] = 0

    hp = hp / (len(img_path) - 1)

    ps, v = [] , []
    bw = hp > tv
    bw_label = label(bw)
    props = regionprops(bw_label, hp)
    for prop in props:
        center_pos = prop['centroid']
        prop_img = prop['intensity_image']
        if np.sum(prop_img) > sz:
            ps.append(center_pos)
            v.append(np.sum(prop_img))

    ps_sel = ps
    v_sel = v
    #hp_smooth = skif.gaussian_filter(hp, sigma=0.4)
    #kernel = gen_filter(radius)
    #hp_conv = convolve(hp, kernel)
    #print "@@@@", np.max(hp_smooth)

    #ps = peak_local_max(hp_smooth, min_distance=mindis)
    #v = [hp_smooth[ps_i[0], ps_i[1]] for ps_i in ps]

    #ps_sel = [ps[i] for i, v_i in enumerate(v) if v_i > tv]
    #v_sel = [v_i for v_i in v if v_i > tv]


    output_name = '%s/%s.csv'%(result_folder, img_path)
    get_a_file(output_name, None)
    with open(output_name , 'w') as f:
        for ps_sel_i, v_sel_i in zip(ps_sel, v_sel):
            f.write('%d %d %.2f\n'%(ps_sel_i[0], ps_sel_i[1], v_sel_i))

    return (img_ID, ps_sel, v_sel)

def get_result(pts, v, gt_pts, sz):
    def get_min_dis(pt, gt_pts):
        mdis, midx = np.Inf, 0
        for gpt_ID, gpt in enumerate(gt_pts):
            disv = distance.euclidean(pt, gpt)
            if disv < mdis:
                mdis = disv
                midx = gpt_ID
        return mdis, midx

    TPs_visual, FPs_visual = [], []
    if len(gt_pts) == 0:
        TP = 0
        FP = len(pts)
        FN = 0
        UC = 0
        for pt, v_i in zip(pts, v):
            FPs_visual.append('-1 %d %d %.2f'%(pt[0], pt[1], v_i))
    else:
        TP, FP, FN, UC = 0 ,0 , 0, 0
        msk = np.zeros((len(gt_pts), 1))
        for pt, v_i in zip(pts, v):
            mdis, midx = get_min_dis(pt, gt_pts)
            if mdis <= sz:
                if msk[midx] == 0: # it is a new TP
                    TP += 1
                    msk[midx] = 1
                    TPs_visual.append('%d %d %d %.2f'%(midx, pt[0], pt[1], v_i))
                else: # it is a duplicated TP
                    UC += 1
                    TPs_visual.append('%d %d %d %.2f'%(midx, pt[0], pt[1], v_i))
            else:
                FP += 1
                FPs_visual.append('-1 %d %d %.2f'%(pt[0], pt[1], v_i))
        FN = len(gt_pts) - np.sum(msk)
    for TP_visual in TPs_visual:
        print TP_visual
    for FP_visual in FPs_visual:
        print FP_visual
    return (TP, FP, FN, UC)

tv = 0.8
sz = 20
radius = 4
mindis = 4
multiprocess = 20
N = 295

result_folder = 'RESULT'

params = [(img_ID, img_path, tv, sz, radius, mindis, result_folder) for img_ID, img_path in enumerate(img_paths)]
pool = Pool(multiprocess)

if 1:
    results = pool.map(get_position, params[:N])
else:
    results = []
    for param in params[:N]:
        result = get_position(param)
        results.append(result)

results_map = {}
for result in results:
    results_map[result[0]] = (result[1], result[2])
print results_map.keys()

TPs, FPs, FNs = 0, 0, 0
for img_ID, img_path in enumerate(img_paths[:N]):
    grd_path = grd_paths[img_ID]
    grd_pts = load_pts(grd_path)
    pts, v = results_map[img_ID]
    TP, FP, FN, UC = get_result(pts, v, grd_pts, sz)
    TPs += TP
    FPs += FP
    FNs += FN
    print img_path, grd_path
    print "\tTP=%d FP=%d FN=%d UC=%d GN=%d"%(TP, FP, FN, UC, len(grd_pts))

precision = TPs / float(TPs + FPs)
recall = TPs / float(TPs + FNs)
f1score = (2.0 * precision * recall) / (precision + recall)
print "t=%.2f s=%d TPs=%d FPs=%d FNs=%d"%(tv, sz, TPs, FPs, FNs)
print precision, recall, f1score
