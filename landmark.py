import cv2
import pickle
import numpy as np
from scipy import io
import mediapipe as mp
from config import cfg

from helpers import relative


mp_face_mesh = mp.solutions.face_mesh  # initialize the face mesh model
face_mesh = mp_face_mesh.FaceMesh(
                max_num_faces=1,  # number of faces to track in each frame
                refine_landmarks=True,  # includes iris landmarks in the face mesh model
                min_detection_confidence=0.5,
                min_tracking_confidence=0.5)

eye_contour_dict = io.loadmat(cfg.eye_contour_2d_idx_path)


def detect_lmk_2d(image, lmk_idx_dict, cfg):
    lmk_num = len(lmk_idx_dict.keys())

    exist_lmk = False
    lmk_2d = np.zeros((lmk_num, 2))
    Pupil_2d = np.zeros((2, 2))
    Eye_contour_2d = np.zeros((2, 16, 2))  # num of lmk on left / right contour are both 16

    if cfg.face_2d_model == 'MediaPipe':
        image.flags.writeable = False
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        mp_results = face_mesh.process(image_rgb)

        if mp_results.multi_face_landmarks:
            exist_lmk = True
            mp_lmk = mp_results.multi_face_landmarks[0]

            Pupil_2d[0, :] = relative(mp_lmk.landmark[cfg.LeftPupil_2d_idx], image.shape)
            Pupil_2d[1, :] = relative(mp_lmk.landmark[cfg.RightPupil_2d_idx], image.shape)
            Pupil_2d = Pupil_2d.astype(float)

            for lmk_name in lmk_idx_dict.keys():
                lmk_2d[lmk_idx_dict[lmk_name]['lmk_idx'], :] = relative(mp_lmk.landmark[lmk_idx_dict[lmk_name]['2d_idx']], image.shape)
            lmk_2d = lmk_2d.astype(float)

            if cfg.eye_mask_type == 'contour':
                LeftEye_contour_idx = eye_contour_dict['LeftEye_mask']
                RightEye_contour_idx = eye_contour_dict['RightEye_mask']

                for i in range(16):
                    Eye_contour_2d[0][i] = relative(mp_lmk.landmark[LeftEye_contour_idx[0][i]], image.shape)
                    Eye_contour_2d[1][i] = relative(mp_lmk.landmark[RightEye_contour_idx[0][i]], image.shape)
                Eye_contour_2d = Eye_contour_2d.astype(int)

        else:
            print('No landmark detected.')

    if exist_lmk:
        lmk_concat = np.concatenate((lmk_2d, Pupil_2d, Eye_contour_2d[0], Eye_contour_2d[1]))
        if (np.min(lmk_concat[:, 0]) >= 0) & (np.max(lmk_concat[:, 0]) < image.shape[1]) & \
                (np.min(lmk_concat[:, 1]) >= 0) & (np.max(lmk_concat[:, 1]) < image.shape[0]):
            pass
        else:
            exist_lmk = False
            print('Exist landmark out of image.')

    return lmk_2d, Pupil_2d, Eye_contour_2d, exist_lmk


def detect_lmk_3d(fs_3d, lmk_idx_dict):
    lmk_num = len(lmk_idx_dict.keys())
    lmk_3d = np.zeros((lmk_num, 3))

    for lmk_name in lmk_idx_dict.keys():
        lmk_3d[lmk_idx_dict[lmk_name]['lmk_idx'], :] = fs_3d[lmk_idx_dict[lmk_name]['3d_idx']]
    lmk_3d = lmk_3d.astype('float32')

    return lmk_3d


