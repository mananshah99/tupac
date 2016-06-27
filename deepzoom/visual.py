#!/usr/bin/env python
#
# call with (in results folder)
# -----------------------------
# python ../../deepzoom/visual.py ../../../data/TrainingData/small_images-level-2 --result tumor_06-21-16 -l 10.35.73.9 -p 5100 --debug
#
# deepzoom_multiserver - Example web application for viewing multiple slides
#
# Copyright (c) 2010-2015 Carnegie Mellon University
#
# This library is free software; you can redistribute it and/or modify it
# under the terms of version 2.1 of the GNU Lesser General Public License
# as published by the Free Software Foundation.
#
# This library is distributed in the hope that it will be useful, but
# WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY
# or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU Lesser General Public
# License for more details.
#
# You should have received a copy of the GNU Lesser General Public License
# along with this library; if not, write to the Free Software Foundation,
# Inc., 51 Franklin Street, Fifth Floor, Boston, MA 02110-1301 USA.
#

from collections import OrderedDict
from flask import Flask, abort, make_response, render_template, url_for
from io import BytesIO
import openslide
from openslide import OpenSlide, OpenSlideError, ImageSlide
from openslide.deepzoom import DeepZoomGenerator
import os
from optparse import OptionParser
from threading import Lock

from PIL import Image
Image.MAX_IMAGE_PIXELS = None

SLIDE_DIR = '.'
SLIDE_CACHE_SIZE = 5
DEEPZOOM_FORMAT = 'jpeg'
DEEPZOOM_TILE_SIZE = 256
DEEPZOOM_OVERLAP = 1
DEEPZOOM_LIMIT_BOUNDS = True
DEEPZOOM_TILE_QUALITY = 80

app = Flask(__name__)
app.config.from_object(__name__)
app.config.from_envvar('DEEPZOOM_MULTISERVER_SETTINGS', silent=True)

import skimage.io as skio
import cv2
import numpy as np
from PIL import Image

def get_osr(path):
    def get_hp():
        ROOT = app.config['HP_DIR']
        slide_name = path.split('/')[-1].split('.')[0]
        '''
        slide_type, slide_id = slide_name.strip().split('_')
        slide_id = int(slide_id)
        hp_path = ""
        if slide_type == 'Tumor':
            if slide_id in range(1, 60+1) or slide_id in range(71, 100+1):
                hp_path = '%s/tr/%s.png'%(ROOT, slide_name)
            else:
                hp_path = '%s/te/%s.png'%(ROOT, slide_name)
        elif slide_type == 'Normal':
            if slide_id in range(1, 90+1) or slide_id in range(101, 150+1):
                hp_path = '%s/tr/%s.png'%(ROOT, slide_name)
            else:
                hp_path = '%s/te/%s.png'%(ROOT, slide_name)
        elif slide_type == 'Test':
                hp_path = '%s/re/%s.png'%(ROOT, slide_name)
        '''
        try:
            hp_path = '%s/%s.png'%(ROOT, slide_name)
            hp = cv2.imread(hp_path, cv2.IMREAD_GRAYSCALE)
            return hp, hp_path
        except Exception as e:
            print e
            return None, None
    ##
    print "Loading image ori: ", path
    img_ori = cv2.imread(path)

    ##
    hp, hp_name = get_hp()
    if hp is None or hp_name is None:
        img = img_ori
    else:
        print "Loading mask image: ", hp_name
        hp_jet = cv2.applyColorMap(hp, cv2.COLORMAP_JET)
        alpha = 0.5
        img = cv2.addWeighted(img_ori, alpha, hp_jet, 1 - alpha, 1);
        img = img[:, :, (2, 1, 0)] # b,g,r -> r,g,b

    img_output = Image.fromarray(img)
    osr = ImageSlide(img_output)
    return osr

def is_image(path):
    for s in ['png', 'jpg', 'jpeg']:
        if path.endswith(s):
            return True
    return False

class PILBytesIO(BytesIO):
    def fileno(self):
        '''Classic PIL doesn't understand io.UnsupportedOperation.'''
        raise AttributeError('Not supported')

class _SlideCache(object):
    def __init__(self, cache_size, dz_opts):
        self.cache_size = cache_size
        self.dz_opts = dz_opts
        self._lock = Lock()
        self._cache = OrderedDict()

    def get(self, path):
        with self._lock:
            if path in self._cache:
                # Move to end of LRU
                slide = self._cache.pop(path)
                self._cache[path] = slide
                return slide

        if is_image(path):
            #osr = ImageSlide(path)
            osr = get_osr(path)
        else:
            osr = OpenSlide(path)
        slide = DeepZoomGenerator(osr, **self.dz_opts)
        try:
            mpp_x = osr.properties[openslide.PROPERTY_NAME_MPP_X]
            mpp_y = osr.properties[openslide.PROPERTY_NAME_MPP_Y]
            slide.mpp = (float(mpp_x) + float(mpp_y)) / 2
        except (KeyError, ValueError):
            slide.mpp = 0

        with self._lock:
            if path not in self._cache:
                if len(self._cache) == self.cache_size:
                    self._cache.popitem(last=False)
                self._cache[path] = slide
        return slide


class _Directory(object):
    def __init__(self, basedir, relpath=''):
        self.name = os.path.basename(relpath)
        self.children = []
        for name in sorted(os.listdir(os.path.join(basedir, relpath))):
            cur_relpath = os.path.join(relpath, name)
            cur_path = os.path.join(basedir, cur_relpath)
            if os.path.isdir(cur_path):
                cur_dir = _Directory(basedir, cur_relpath)
                if cur_dir.children:
                    self.children.append(cur_dir)
            elif is_image(cur_path):
                self.children.append(_SlideFile(cur_relpath))
            elif OpenSlide.detect_format(cur_path):
                self.children.append(_SlideFile(cur_relpath))


class _SlideFile(object):
    def __init__(self, relpath):
        self.name = os.path.basename(relpath)
        self.url_path = relpath


@app.before_first_request
def _setup():
    app.basedir = os.path.abspath(app.config['SLIDE_DIR'])
    config_map = {
        'DEEPZOOM_TILE_SIZE': 'tile_size',
        'DEEPZOOM_OVERLAP': 'overlap',
        'DEEPZOOM_LIMIT_BOUNDS': 'limit_bounds',
    }
    opts = dict((v, app.config[k]) for k, v in config_map.items())
    app.cache = _SlideCache(app.config['SLIDE_CACHE_SIZE'], opts)


def _get_slide(path):
    path = os.path.abspath(os.path.join(app.basedir, path))
    if not path.startswith(app.basedir + os.path.sep):
        # Directory traversal
        abort(404)
    if not os.path.exists(path):
        abort(404)
    try:
        slide = app.cache.get(path)
        slide.filename = os.path.basename(path)
        return slide
    except OpenSlideError:
        abort(404)


@app.route('/')
def index():
    return render_template('files.html', root_dir=_Directory(app.basedir), page_name = app.config['PAGE_NAME'])


@app.route('/<path:path>')
def slide(path):
    slide = _get_slide(path)
    slide_url = url_for('dzi', path=path)
    return render_template('slide-fullpage.html', slide_url=slide_url, slide_filename=slide.filename, slide_mpp=slide.mpp)


@app.route('/<path:path>.dzi')
def dzi(path):
    slide = _get_slide(path)
    format = app.config['DEEPZOOM_FORMAT']
    resp = make_response(slide.get_dzi(format))
    resp.mimetype = 'application/xml'
    return resp


@app.route('/<path:path>_files/<int:level>/<int:col>_<int:row>.<format>')
def tile(path, level, col, row, format):
    slide = _get_slide(path)
    format = format.lower()
    if format != 'jpeg' and format != 'png':
        # Not supported by Deep Zoom
        abort(404)
    try:
        tile = slide.get_tile(level, (col, row))
    except ValueError:
        # Invalid level or coordinates
        abort(404)
    buf = PILBytesIO()
    tile.save(buf, format, quality=app.config['DEEPZOOM_TILE_QUALITY'])
    resp = make_response(buf.getvalue())
    resp.mimetype = 'image/%s' % format
    return resp


if __name__ == '__main__':
    parser = OptionParser(usage='Usage: %prog [options] [slide-directory]')
    parser.add_option('-B', '--ignore-bounds', dest='DEEPZOOM_LIMIT_BOUNDS',
                default=True, action='store_false',
                help='display entire scan area')
    parser.add_option('-c', '--config', metavar='FILE', dest='config',
                help='config file')
    parser.add_option('-d', '--debug', dest='DEBUG', action='store_true',
                help='run in debugging mode (insecure)')
    parser.add_option('-e', '--overlap', metavar='PIXELS',
                dest='DEEPZOOM_OVERLAP', type='int',
                help='overlap of adjacent tiles [1]')
    parser.add_option('-f', '--format', metavar='{jpeg|png}',
                dest='DEEPZOOM_FORMAT',
                help='image format for tiles [jpeg]')
    parser.add_option('-l', '--listen', metavar='ADDRESS', dest='host',
                default='127.0.0.1',
                help='address to listen on [127.0.0.1]')
    parser.add_option('-p', '--port', metavar='PORT', dest='port', type='int', default=5000, help='port to listen on [5000]')
    parser.add_option('-Q', '--quality', metavar='QUALITY',
                dest='DEEPZOOM_TILE_QUALITY', type='int',
                help='JPEG compression quality [75]')
    parser.add_option('-s', '--size', metavar='PIXELS',
                dest='DEEPZOOM_TILE_SIZE', type='int',
                help='tile size [256]')
    parser.add_option('-n', '--name',   default='Slides', dest='page_name', type='str', help='Web Page Name')
    parser.add_option('-r', '--result', metavar="RESULT_FOLDER", dest='result_folder', type='str', default='', help='Result folder')

    (opts, args) = parser.parse_args()
    # Load config file if specified
    if opts.config is not None:
        app.config.from_pyfile(opts.config)
    # Overwrite only those settings specified on the command line
    for k in dir(opts):
        if not k.startswith('_') and getattr(opts, k) is None:
            delattr(opts, k)
    app.config.from_object(opts)
    # Set slide directory
    try:
        app.config['SLIDE_DIR'] = args[0] # the image folder
        app.config['HP_DIR'] = opts.result_folder # the heatmap folder
        app.config['PAGE_NAME'] = opts.page_name
    except IndexError:
        pass

    app.run(host=opts.host, port=opts.port, threaded=True)
