import yaml
import sys

sys.path.append('/caffe-alrnet/')
import caffe

from DataLoader import *
from Augmentations import *

class MyDataLayer(caffe.Layer):

    def setup(self, bottom, top):
        params_str = self.param_str.split(',')
        params = [yaml.load(item) for item in params_str]

        self.source_dir = params[0]['source_dir']
        self.target_size_w = params[1]['target_size_w']
        self.target_size_h = params[2]['target_size_h']
        self.batch_size = params[3]['batch_size']
        self.stage = params[4]['stage']

        augmentations = Compose([RandomHorizontallyFlip(),RandomRotate(),AdjustBrightness(),AdjustContrast()])
        self.augmentations = augmentations

        self.batch_loader = DataLoader(source_dir=self.source_dir,
                                              batch_size=self.batch_size,
                                              target_size_w=self.target_size_w,
                                              target_size_h=self.target_size_h,
                                              stage=self.stage,
                                              augmentations=self.augmentations)

        # N * C * W * H
        top[0].reshape(self.batch_size, 3, self.target_size_w, self.target_size_h)  # RGB

        if self.stage == 'train': 
            top[1].reshape(self.batch_size, 1, self.target_size_w, self.target_size_h)  # Label
            top[2].reshape(self.batch_size, 1, self.target_size_w, self.target_size_h)  # Label_weight

    def reshape(self, bottom, top):
        pass

    def forward(self, bottom, top):
       if self.stage == 'train': 
            imgs_RGB, imgs_gt, imgs_gt_weight = self.batch_loader.next_train_batch()
            top[0].data[...] = imgs_RGB
            top[1].data[...] = imgs_gt
            top[2].data[...] = imgs_gt_weight
       else:
            imgs_RGB= self.batch_loader.next_test_batch()
            top[0].data[...] = imgs_RGB

    def backward(self, top, propagate_down, bottom):
        pass

