# Adapted from https://github.com/ZijunDeng/pytorch-semantic-segmentation/blob/master/utils/joint_transforms.py

import math
import numbers
import random
import numpy as np
import scipy.io
from PIL import Image, ImageOps, ImageEnhance

class Compose(object):
    def __init__(self, augmentations):
        self.augmentations = augmentations

    def __call__(self, array_RGB,array_gt):

        img_RGB = Image.fromarray(array_RGB,mode='RGB')
        img_gt = Image.fromarray(array_gt)

        for a in self.augmentations:
            img_RGB, img_gt = a(img_RGB, img_gt)

        return np.array(img_RGB), np.array(img_gt)

class RandomHorizontallyFlip(object):
    def __call__(self, img_RGB, img_gt):
        if random.random() <= 0.5:
            return img_RGB.transpose(Image.FLIP_LEFT_RIGHT), img_gt.transpose(Image.FLIP_LEFT_RIGHT)
        return img_RGB, img_gt

class RandomRotate(object):
    def __call__(self, img_RGB, img_gt):
        rand= random.random()
        if rand <= 0.25:
            return img_RGB.rotate(90), img_gt.rotate(90)
        elif rand <= 0.50:
            return img_RGB.rotate(180), img_gt.rotate(180)
        elif rand <= 0.75:
            return img_RGB.rotate(270), img_gt.rotate(270)
        else:
            return img_RGB, img_gt

class AdjustBrightness(object):
    def __call__(self, img_RGB, img_gt):
        rand= random.random()
        enhancer = ImageEnhance.Brightness(img_RGB)
        if rand <= 0.33:
            return enhancer.enhance(1.3), img_gt
        elif rand <= 0.66:
            return enhancer.enhance(0.7), img_gt
        else:
            return img_RGB, img_gt

class AdjustContrast(object):
    def __call__(self, img_RGB, img_gt):
        rand= random.random()
        enhancer = ImageEnhance.Contrast(img_RGB)
        if rand <= 0.33:
            return enhancer.enhance(1.3), img_gt
        elif rand <= 0.66:
            return enhancer.enhance(0.7), img_gt
        else:
            return img_RGB, img_gt

