#! /usr/bin/env python
# coding=utf-8
from easydict import EasyDict as edict


__C                           = edict()
# Consumers can get config by: from config import cfg

cfg                           = __C

# iou threshold
__C.iou                       = 0.45
# score threshold
__C.score                     = 0.25
# resize images to
__C.size                      = 416
# codec used in VideoWriter when saving video to file
__C.output_format             = 'XVID'
# path to weights file
__C.weights                   = './yolov4-416'

#==========================================================
# YOLO options
__C.YOLO                      = edict()

__C.YOLO.CLASSES              = "./data/classes/coco.names"
__C.YOLO.ANCHORS              = [12,16, 19,36, 40,28, 36,75, 76,55, 72,146, 142,110, 192,243, 459,401]
__C.YOLO.STRIDES              = [8, 16, 32]
__C.YOLO.XYSCALE              = [1.2, 1.1, 1.05]
