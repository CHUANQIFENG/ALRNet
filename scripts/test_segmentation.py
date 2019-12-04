#!/usr/bin/python
# -*- coding:utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
import os.path
import json
import glob
import scipy
import argparse
import scipy.io
import math
from cv2 import imread, imwrite
caffe_root = '/caffe-alrnet/' 			
import sys
sys.path.insert(0, caffe_root + 'python')

import caffe

# Import arguments
parser = argparse.ArgumentParser()
parser.add_argument('--model', type=str, required=True)
parser.add_argument('--weights', type=str, required=True)
parser.add_argument('--iter', type=int, required=True)
parser.add_argument('--file_dir', type=str, required=True)
parser.add_argument('--out_dir', type=str, required=True)
args = parser.parse_args()

caffe.set_mode_gpu()
caffe.set_device(0)

net = caffe.Net(args.model,
                args.weights,
                caffe.TEST)

dir_RGB = args.file_dir
file_list = glob.glob(dir_RGB + '/*.png')

for i in range(0, args.iter):

    print(i)

	net.forward()

    item = file_list[i]
    item_name = os.path.basename(item)

	predicted = net.blobs['prob'].data
	output = np.squeeze(predicted[0,:,:,:])

    scipy.io.savemat(args.out_dir+'/'+item_name.replace('.png','.mat'), mdict={'image_sub': output}, oned_as='row')

print 'Success!'

