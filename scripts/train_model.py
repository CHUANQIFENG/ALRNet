#!/usr/bin/python
# -*- coding:utf-8 -*-

caffe_root = '/caffe-alrnet/'
import sys#
sys.path.insert(0, caffe_root + 'python')

import caffe
import numpy as np

caffe.set_mode_gpu()
caffe.set_device(0)
solver= caffe.get_solver("/ALRNet/models/potsdam_solver.prototxt")

weights = "/ALRNet/models/PretrainedModels/VGG_ILSVRC_16_layers.caffemodel"
solver.net.copy_from(weights);

solver.solve()
