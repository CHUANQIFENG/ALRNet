import glob
import random
import os.path
import scipy.io
import numpy as np
from PIL import Image

from Augmentations import *

class DataLoader(object):

    def __init__(self, source_dir, batch_size, target_size_w, target_size_h, stage,augmentations):
        self.batch_size = batch_size
        self.target_size_w = target_size_w
        self.target_size_h = target_size_h
        self.stage = stage
        self.augmentations = augmentations

        self.imgs_RGB = np.zeros((self.batch_size,3,self.target_size_w,self.target_size_h))
        self.imgs_gt = np.zeros((self.batch_size,1,self.target_size_w,self.target_size_h))
        self.imgs_gt_weight = np.zeros((self.batch_size,1,self.target_size_w,self.target_size_h))

        if self.stage == 'train': 
            self.dir_RGB = source_dir + '/IRRG'
            self.dir_gt = source_dir + '/Label'
            self.file_list = glob.glob(self.dir_RGB + '/*.png')
            random.shuffle(self.file_list)
        else:
            self.dir_RGB = source_dir + '/IRRG'
            self.file_list = glob.glob(self.dir_RGB + '/*.png')

        self.cursor = 0

    def next_train_batch(self):
        if self.cursor + self.batch_size > len(self.file_list):
            self.cursor = 0
            random.shuffle(self.file_list)

        index = 0
        
        for sub_cursor in range(self.cursor, self.cursor + self.batch_size):
            # Get file name
            item = self.file_list[sub_cursor]
            item_name = os.path.basename(item)
            #print item_name

            # Get file path
            image_path_RGB = self.dir_RGB + '/' + item_name
            image_path_gt = self.dir_gt + '/' + item_name.replace('.png','.mat')

            # For RGB file
            array_RGB = np.array(Image.open(image_path_RGB))

            # For gt file
            mat_gt = scipy.io.loadmat(image_path_gt)['image_sub']
            array_gt=np.asarray(mat_gt)

            # Augmentation
            array_RGB,array_gt = self.augmentations(array_RGB,array_gt)

            # For gt_weight file
            array_gt_weight=np.where(array_gt==0, 1 , np.where(array_gt==1, 1 , np.where(array_gt==2, 1 , np.where(array_gt==3, 1 , np.where(array_gt==4, 1 , 1)))))
            
            array_RGB = array_RGB.transpose((2, 0, 1))
            array_gt = np.expand_dims(array_gt, axis=0)
            array_gt_weight = np.expand_dims(array_gt_weight, axis=0)

            self.imgs_RGB[index,...] = array_RGB
            self.imgs_gt[index, ...] = array_gt
            self.imgs_gt_weight[index, ...] = array_gt_weight

            index=index+1

        self.cursor += self.batch_size
        return self.imgs_RGB, self.imgs_gt, self.imgs_gt_weight

    def next_test_batch(self):
        if self.cursor == len(self.file_list):
            self.cursor = 0

        index = 0
        
        for sub_cursor in range(self.cursor, self.cursor + self.batch_size):
            # Get file name
            item = self.file_list[sub_cursor]
            item_name = os.path.basename(item)
            #print item_name

            # Get file path
            image_path_RGB = self.dir_RGB + '/' + item_name

            # For RGB file
            array_RGB = np.array(Image.open(image_path_RGB))
            
            array_RGB = array_RGB.transpose((2, 0, 1))

            self.imgs_RGB[index,...] = array_RGB

            index=index+1

        self.cursor += self.batch_size
        return self.imgs_RGB

