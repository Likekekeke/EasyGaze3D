# Use MATLAB to calibrate the camera
import os
import numpy as np
from scipy import io
from config import cfg

camera_matrix = np.array([
                    [1472.4642, 0, 757.8711],
                    [0, 1474.3204, 342.6025],
                    [0, 0, 1]])

dist_coeffs = np.array([
                    [-0.1279],
                    [0.3522],
                    [0],
                    [0],
])

intrinsic_params = {'camera_matrix': camera_matrix,
                    'dist_coeffs': dist_coeffs}

subj_idx = cfg.EasyCali.subj_idx
image_save_dir = os.path.join(cfg.EasyCali.image_save_dir, 'subject_{}'.format(subj_idx))
if not os.path.exists(image_save_dir):
    os.makedirs(image_save_dir)

io.savemat(os.path.join(image_save_dir, 'intrinsic_params.mat'), mdict=intrinsic_params)













