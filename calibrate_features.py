import os
import sys
import cv2
import time
import pickle
import numpy as np
from scipy import io
import matplotlib.pyplot as plt

from config import cfg
from facial_shape import recon_facial_shape
from landmark import detect_lmk_2d, detect_lmk_3d
from helpers import relative, head_pose, eye_mask, pupil_center, eyeball_center, average_gaze, lines_intersection, W_2_I, W_2_C, I_2_W
from visualization import visualize_3d

sys.path.append('./TDDFA_V2/')


if __name__ == '__main__':
    subj_idx = cfg.EasyCali.subj_idx

    image_save_dir = os.path.join('./calibration/', cfg.EasyCali.image_save_dir, 'subject_{}'.format(subj_idx))
    image_name_lst = os.listdir(image_save_dir)
    if 'intrinsic_params.mat' in image_name_lst:
        image_name_lst.remove('intrinsic_params.mat')
    image_name_lst.sort()
    print('Now:', len(image_name_lst))

    intrinsic_params = io.loadmat(os.path.join(image_save_dir, 'intrinsic_params.mat'))

    with open(cfg.lmk_idx_path, 'rb') as lmk_idx_file:
        lmk_idx_dict = pickle.load(lmk_idx_file)

    result_save_path = os.path.join('./calibration/', cfg.EasyCali.result_save_dir, 'subject_{}'.format(subj_idx))
    if not os.path.exists(result_save_path):
        os.makedirs(result_save_path)

    # Facial shape recovery
    fs_3d_path = os.path.join(result_save_path, 'facial_shape_3d.npy')
    lmk_3d_path = os.path.join(result_save_path, 'landmark_3d.npy')

    if not os.path.exists(fs_3d_path) or not os.path.exists(lmk_3d_path):
        assert cfg.face_3d_model in ['3DDFA_V2']
        all_fs_3d = np.zeros((len(image_name_lst), cfg.face_3d_model_shape[0], 3))

        for image_idx in range(len(image_name_lst)):
            image_name = image_name_lst[image_idx]
            image_path = os.path.join(image_save_dir, image_name)

            image = cv2.imread(image_path)
            fs_3d, fs_frontal_3d, exist_fs = recon_facial_shape(image, cfg)

            all_fs_3d[image_idx, :, :] = fs_3d
            print(image_name, '-- facial shape done.')

        avg_fs_3d = np.mean(all_fs_3d, axis=0)
        print(avg_fs_3d.shape)
        np.save(fs_3d_path, avg_fs_3d)

        lmk_3d = detect_lmk_3d(avg_fs_3d, lmk_idx_dict)
        np.save(lmk_3d_path, lmk_3d)

    else:
        avg_fs_3d = np.load(fs_3d_path)
        lmk_3d = np.load(lmk_3d_path)

    fs_3d, fs_frontal_3d = avg_fs_3d, avg_fs_3d[: cfg.face_3d_model_frontal_idx, :]

    # Eyeball center calibration
    EyeballCenter_path = os.path.join(result_save_path, 'eyeball_center.npy')
    gaze_lines_path = os.path.join(result_save_path, 'gaze_lines.mat')

    if not os.path.exists(EyeballCenter_path) or not os.path.exists(gaze_lines_path):

        left_lines_strt = np.zeros((len(image_name_lst), 3))
        left_lines_vec = np.zeros((len(image_name_lst), 3))
        right_lines_strt = np.zeros((len(image_name_lst), 3))
        right_lines_vec = np.zeros((len(image_name_lst), 3))

        no_lmk_idx = []

        for image_idx in range(len(image_name_lst)):
            image_name = image_name_lst[image_idx]
            image_path = os.path.join(image_save_dir, image_name)

            image = cv2.imread(image_path)

            # 2d landmark detection
            assert cfg.face_2d_model in ['MediaPipe']
            lmk_2d, Pupil_2d, Eye_contour_2d, exist_lmk = detect_lmk_2d(image, lmk_idx_dict, cfg)

            if not exist_lmk:
                no_lmk_idx.append(image_idx)
                continue

            # Head pose
            rot_v, rot_mat, trans_v = head_pose(lmk_2d, lmk_3d, intrinsic_params)

            # W ---> I
            fs_W, fs_frontal_W = fs_3d, fs_frontal_3d
            fs_frontal_I = W_2_I(fs_frontal_W, rot_v, trans_v, intrinsic_params)

            # Two points for one gaze line
            Pupil_W_1 = I_2_W(Pupil_2d, rot_mat, trans_v, intrinsic_params, scalar=3e5)
            Pupil_W_2 = I_2_W(Pupil_2d, rot_mat, trans_v, intrinsic_params, scalar=6e5)

            left_lines_strt[image_idx, :] = Pupil_W_1[0]
            left_lines_vec[image_idx, :] = Pupil_W_2[0] - Pupil_W_1[0]
            right_lines_strt[image_idx, :] = Pupil_W_1[1]
            right_lines_vec[image_idx, :] = Pupil_W_2[1] - Pupil_W_1[1]

            print(image_name, '-- gaze line done.')

        left_lines_strt = np.delete(left_lines_strt, no_lmk_idx, axis=0)
        left_lines_vec = np.delete(left_lines_vec, no_lmk_idx, axis=0)
        right_lines_strt = np.delete(right_lines_strt, no_lmk_idx, axis=0)
        right_lines_vec = np.delete(right_lines_vec, no_lmk_idx, axis=0)

        LeftEyeballCenter_W = lines_intersection(left_lines_strt, left_lines_vec)
        RightEyeballCenter_W = lines_intersection(right_lines_strt, right_lines_vec)

        EyeballCenter_W = np.array([LeftEyeballCenter_W, RightEyeballCenter_W])
        np.save(EyeballCenter_path, EyeballCenter_W)

        gaze_lines_dict = {
            'left_lines_strt': left_lines_strt,
            'left_lines_vec': left_lines_vec,
            'right_lines_strt': right_lines_strt,
            'right_lines_vec': right_lines_vec,
        }
        io.savemat(gaze_lines_path, mdict=gaze_lines_dict)

    else:
        EyeballCenter_W = np.load(EyeballCenter_path)
        LeftEyeballCenter_W, RightEyeballCenter_W = EyeballCenter_W[0], EyeballCenter_W[1]

        gaze_lines_dict = io.loadmat(gaze_lines_path)

    # Visualize
    if cfg.EasyCali.visualize:
        vis_3d_dict_W = {  # WCS default
            'CS': 'W',
            'view': (90, -90),

            'facial_shape_3d': {'obj': fs_3d, 's': 20, 'alpha': 0.05},
            'LeftEyeballCenter': {'obj': LeftEyeballCenter_W, 'color': 'g', 's': 40, 'alpha': 0.8},
            'RightEyeballCenter': {'obj': RightEyeballCenter_W, 'color': 'g', 's': 40, 'alpha': 0.8},
            'landmark': {'obj': lmk_3d, 'color': 'r', 's': 40, 'alpha': 0.8},
            'gaze_lines_dict': {'obj': gaze_lines_dict, 'sampling_rate': 1/5, 'start_pos': 0.3, 'end_pos': 0.7,
                                'left_color': 'r', 'right_color': 'b', 'linewidth': 2.0, 'alpha': 0.4},

            'show_tick': True,  # [True, False]
        }

        visualize_3d(vis_3d_dict_W)









