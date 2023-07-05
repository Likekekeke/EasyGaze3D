import os
import sys
import cv2
import pickle
import numpy as np
from scipy import io
import mediapipe as mp
import matplotlib.pyplot as plt

from landmark import detect_lmk_2d, detect_lmk_3d
from facial_shape import recon_facial_shape
from helpers import relative, head_pose, eye_mask, pupil_center, eyeball_center, average_gaze, W_2_I, W_2_C
from visualization import visualize_3d

from config import cfg

sys.path.append('./TDDFA_V2/')

with open(cfg.lmk_idx_path, 'rb') as lmk_idx_file:
    lmk_idx_dict = pickle.load(lmk_idx_file)


def gaze_vector(image, intrinsic_params, cfg, saved_fs_3d=None, saved_EyeballCenter_W=None):
    """
    :param image: cv2.imread(image_path)
    :param intrinsic_params: {'camera_matrix': 3 * 3, 'dist_coeffs': 5 * 1}
    :param cfg:
    :param saved_fs_3d:
    :param saved_EyeballCenter_W:
    :return:
    """

    use_EasyCali = cfg.use_EasyCali

    # 2d landmark detection
    assert cfg.face_2d_model in ['MediaPipe']
    lmk_2d, Pupil_2d, Eye_contour_2d, exist_lmk = detect_lmk_2d(image, lmk_idx_dict, cfg)

    if not exist_lmk:
        return np.zeros((1, 3)).squeeze(), np.zeros((1, 3)).squeeze(), {}, False

    # 3d facial shape reconstruction
    if not use_EasyCali:
        if cfg.use_saved_3d_model and cfg.face_3d_model == '3DDFA_V2':
            assert saved_fs_3d is not None
            fs_3d, fs_frontal_3d, exist_fs = saved_fs_3d, saved_fs_3d[: cfg.face_3d_model_frontal_idx, :], True
        else:
            assert cfg.face_3d_model in ['3DDFA_V2']
            fs_3d, fs_frontal_3d, exist_fs = recon_facial_shape(image, cfg)
    else:  # Use the facial shape recovered by EasyCali module
        assert saved_fs_3d is not None
        fs_3d, fs_frontal_3d, exist_fs = saved_fs_3d, saved_fs_3d[: cfg.face_3d_model_frontal_idx, :], True

    if not exist_fs:
        return np.zeros((1, 3)).squeeze(), np.zeros((1, 3)).squeeze(), {}, False

    # 3d landmark detection
    lmk_3d = detect_lmk_3d(fs_3d, lmk_idx_dict)

    # Head pose
    rot_v, rot_mat, trans_v = head_pose(lmk_2d, lmk_3d, intrinsic_params)

    # W ---> I
    fs_W, fs_frontal_W = fs_3d, fs_frontal_3d
    fs_frontal_I = W_2_I(fs_frontal_W, rot_v, trans_v, intrinsic_params)

    # Eye mask in W
    LeftEye_mask_W, RightEye_mask_W = eye_mask(Eye_contour_2d, fs_frontal_I, fs_frontal_W, image.shape, cfg)
    if LeftEye_mask_W.shape[0] <= 10 or RightEye_mask_W.shape[0] <= 10:
        print('Eye mask too small.')
        return np.zeros((1, 3)).squeeze(), np.zeros((1, 3)).squeeze(), {}, False

    # Pupil center in W
    LeftPupil_W, RightPupil_W = pupil_center(Pupil_2d, LeftEye_mask_W, RightEye_mask_W, rot_v, trans_v, intrinsic_params)

    # Eyeball center in W
    if not use_EasyCali:  # spherical fitting
        LeftEyeballCenter_W, RightEyeballCenter_W = eyeball_center(LeftEye_mask_W, RightEye_mask_W)
    else:
        assert saved_EyeballCenter_W is not None
        LeftEyeballCenter_W, RightEyeballCenter_W = saved_EyeballCenter_W

    # W to C
    LeftPupil_C, RightPupil_C = W_2_C(LeftPupil_W, rot_mat, trans_v), W_2_C(RightPupil_W, rot_mat, trans_v)
    LeftEyeballCenter_C, RightEyeballCenter_C = W_2_C(LeftEyeballCenter_W, rot_mat, trans_v), W_2_C(RightEyeballCenter_W, rot_mat, trans_v)

    # Gaze vector
    gaze_W = average_gaze(LeftPupil_W, RightPupil_W, LeftEyeballCenter_W, RightEyeballCenter_W)
    gaze_C = average_gaze(LeftPupil_C, RightPupil_C, LeftEyeballCenter_C, RightEyeballCenter_C)
    gaze_I = average_gaze(Pupil_2d[0], Pupil_2d[1],
                          W_2_I(LeftEyeballCenter_W, rot_v, trans_v, intrinsic_params), W_2_I(RightEyeballCenter_W, rot_v, trans_v, intrinsic_params))

    # Visualize
    if cfg.visualize:
        fs_C = W_2_C(fs_W, rot_mat, trans_v)
        LeftEye_mask_C = W_2_C(LeftEye_mask_W, rot_mat, trans_v)
        RightEye_mask_C = W_2_C(RightEye_mask_W, rot_mat, trans_v)
        lmk_W = lmk_3d
        lmk_C = W_2_C(lmk_W, rot_mat, trans_v)

        vis_3d_dict_C = {  # CCS visualize default
            'CS': 'C',
            'view': (90, 90),

            'facial_shape_3d': {'obj': fs_C, 's': 20, 'alpha': 0.05},
            # 'LeftEye_mask': {'obj': LeftEye_mask_C, 's': 20, 'alpha': 0.15},
            # 'RightEye_mask': {'obj': RightEye_mask_C, 's': 20, 'alpha': 0.15},
            'LeftPupil': {'obj': LeftPupil_C, 'color': 'b', 's': 40, 'alpha': 0.8},
            'RightPupil': {'obj': RightPupil_C, 'color': 'b', 's': 40, 'alpha': 0.8},
            'LeftEyeballCenter': {'obj': LeftEyeballCenter_C, 'color': 'g', 's': 40, 'alpha': 0.8},
            'RightEyeballCenter': {'obj': RightEyeballCenter_C, 'color': 'g', 's': 40, 'alpha': 0.8},
            'landmark': {'obj': lmk_C, 'color': 'r', 's': 40, 'alpha': 0.8},
            'gaze': {'obj': gaze_C, 'scalar': 5e4, 'LeftPupil': LeftPupil_C, 'RightPupil': RightPupil_C, 'color': 'r', 'linewidth': 4.0, 'alpha': 0.8},

            'show_tick': True,  # [True, False]
        }

        vis_3d_dict_W = {  # WCS visualize default
            'CS': 'W',
            'view': (90, -90),

            'facial_shape_3d': {'obj': fs_W, 's': 20, 'alpha': 0.05},
            # 'LeftEye_mask': {'obj': LeftEye_mask_W, 's': 20, 'alpha': 0.15},
            # 'RightEye_mask': {'obj': RightEye_mask_W, 's': 20, 'alpha': 0.15},
            'LeftPupil': {'obj': LeftPupil_W, 'color': 'b', 's': 40, 'alpha': 0.8},
            'RightPupil': {'obj': RightPupil_W, 'color': 'b', 's': 40, 'alpha': 0.8},
            'LeftEyeballCenter': {'obj': LeftEyeballCenter_W, 'color': 'g', 's': 40, 'alpha': 0.8},
            'RightEyeballCenter': {'obj': RightEyeballCenter_W, 'color': 'g', 's': 40, 'alpha': 0.8},
            'landmark': {'obj': lmk_W, 'color': 'r', 's': 40, 'alpha': 0.8},
            'gaze': {'obj': gaze_W, 'scalar': 5e4, 'LeftPupil': LeftPupil_W, 'RightPupil': RightPupil_W, 'color': 'r', 'linewidth': 4.0, 'alpha': 0.8},

            'show_tick': True,  # [True, False]
        }

        visualize_3d(vis_3d_dict_C)
        visualize_3d(vis_3d_dict_W)

    if cfg.extension:
        dist_to_cam = (LeftPupil_C[2] + RightPupil_C[2]) / 2
        extend_dict = {
            'Pupil_I': Pupil_2d,
            'gaze_I': gaze_I,
            'Pupil_W': np.array([LeftPupil_W, RightPupil_W]),
            'distance_to_camera': dist_to_cam,
        }
    else:
        extend_dict = {}

    if gaze_C[2] > 0:  # TODO: Need to consider the WRONG CCS side about camera position
        gaze_C[2] = -gaze_C[2]
        print('Wrong CCS side by PnP solver.')

    return gaze_C, gaze_W, extend_dict, True


if __name__ == '__main__':
    image = cv2.imread('./examples/02.jpg')

    cfg.use_EasyCali = True
    cfg.use_saved_3d_model = True
    cfg.visualize = True

    subj_idx = cfg.EasyCali.subj_idx
    image_save_dir = os.path.join('./calibration/', cfg.EasyCali.image_save_dir, 'subject_{}'.format(subj_idx))
    intrinsic_params = io.loadmat(os.path.join(image_save_dir, 'intrinsic_params.mat'))

    result_save_path = os.path.join('./calibration/', cfg.EasyCali.result_save_dir, 'subject_{}'.format(subj_idx))
    fs_3d_path = os.path.join(result_save_path, 'facial_shape_3d.npy')
    EyeballCenter_path = os.path.join(result_save_path, 'eyeball_center.npy')

    saved_fs_3d = np.load(fs_3d_path)
    saved_EyeballCenter_W = np.load(EyeballCenter_path)

    gaze_C, gaze_W, extend_dict, _ = gaze_vector(image, intrinsic_params, cfg, saved_fs_3d=saved_fs_3d, saved_EyeballCenter_W=saved_EyeballCenter_W)


