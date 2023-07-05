from yacs.config import CfgNode as CN
import argparse
import yaml
import os
import numpy as np

cfg = CN()

# Recommend not to change
cfg.face_2d_model = 'MediaPipe'
cfg.face_3d_model = '3DDFA_V2'

cfg.lmk_idx_path = './landmark_index.pkl'
cfg.LeftPupil_2d_idx = 473
cfg.RightPupil_2d_idx = 468

cfg.face_3d_model_shape = (38365, 3)
cfg.face_3d_model_frontal_idx = 16237

cfg.eye_mask_3d_idx_path = './eye_mask_3d_index.mat'
cfg.eye_contour_2d_idx_path = './eye_contour_2d_index.mat'

# Settings
cfg.use_EasyCali = False  # [True, False]
cfg.use_saved_3d_model = False  # [True, False]
cfg.eye_mask_type = 'contour'  # ['simple', 'contour']

cfg.visualize = False  # [True, False]
cfg.extension = False  # [True, False]

cfg.render = False  # [True, False]

# Easy-Cali module
cfg.EasyCali = CN()
cfg.EasyCali.image_save_dir = './images/'
cfg.EasyCali.result_save_dir = './results/'
cfg.EasyCali.pure_black_path = './pure_black.jpg'

cfg.EasyCali.subj_idx = '00'
cfg.EasyCali.to_save = False  # [True, False]

cfg.EasyCali.visualize = False  # [True, False]






