# coding: utf-8

__author__ = 'cleardusk'

import os
import sys
import argparse
import cv2
import yaml
import numpy as np
import time

from FaceBoxes import FaceBoxes
from TDDFA_EG3D import TDDFA
from utils.render import render
# from utils.render_ctypes import render  # faster
from utils.depth import depth
from utils.pncc import pncc
from utils.uv import uv_tex
from utils.pose import viz_pose
from utils.serialization import ser_to_ply, ser_to_obj
from utils.functions import draw_landmarks, get_suffix
from utils.tddfa_util import str2bool


def shape(image):
    cfg = yaml.load(open('./TDDFA_V2/configs/mb1_120x120.yml'), Loader=yaml.SafeLoader)

    gpu_mode = 'cpu'
    tddfa = TDDFA(gpu_mode=gpu_mode, **cfg)
    face_boxes = FaceBoxes()

    # Given a still image path and load to BGR channel
    img = image

    # Detect faces, get 3DMM params and roi boxes
    boxes = face_boxes(img)
    n = len(boxes)
    if n == 0:
        print(f'No face detected, exit')
        sys.exit(-1)

    param_lst, roi_box_lst = tddfa(img, boxes)
    shape_lst = tddfa.recon_vers(param_lst, roi_box_lst, dense_flag='3d')

    exist_shape = False
    if len(shape_lst) != 0:
        exist_shape = True

    return shape_lst, exist_shape


def get_render_params(image, tddfa, face_boxes):
    # Given a still image path and load to BGR channel
    img = image

    # Detect faces, get 3DMM params and roi boxes
    boxes = face_boxes(img)
    n = len(boxes)
    if n == 0:
        return [], [], [], 0, False

    param_lst, roi_box_lst = tddfa(img, boxes)
    R_lst, offset_lst, roi_box_lst, size = tddfa.render_params(param_lst, roi_box_lst, dense_flag='3d')

    exist_shape = False
    if len(R_lst) != 0:
        exist_shape = True

    return R_lst, offset_lst, roi_box_lst, size, exist_shape


