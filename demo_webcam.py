import os
import sys
import cv2
import yaml
import time
import pickle
import numpy as np
from scipy import io
import matplotlib.pyplot as plt

from gaze_estimation import gaze_vector
from config import cfg
from helpers import kalman_filter


if __name__ == '__main__':
    # Initialize
    cfg.use_EasyCali = True
    cfg.use_saved_3d_model = True

    cfg.visualize = False
    cfg.extension = True

    # Initialize renderer
    if cfg.render:
        assert cfg.face_3d_model == '3DDFA_V2'

        from TDDFA_V2.utils.render import render
        from TDDFA_V2.TDDFA_EG3D import TDDFA
        from TDDFA_V2.FaceBoxes import FaceBoxes
        from TDDFA_V2.facial_shape import get_render_params
        from TDDFA_V2.utils.tddfa_util import similar_transform

        cfg_tddfa = yaml.load(open('./TDDFA_V2/configs/mb1_120x120.yml'), Loader=yaml.SafeLoader)
        gpu_mode = 'cpu'
        tddfa = TDDFA(gpu_mode=gpu_mode, **cfg_tddfa)
        face_boxes = FaceBoxes()

    # Initialize subject-specific features
    subj_idx = cfg.EasyCali.subj_idx
    image_save_dir = os.path.join('./calibration/', cfg.EasyCali.image_save_dir, 'subject_{}'.format(subj_idx))
    intrinsic_params = io.loadmat(os.path.join(image_save_dir, 'intrinsic_params.mat'))

    result_save_path = os.path.join('./calibration/', cfg.EasyCali.result_save_dir, 'subject_{}'.format(subj_idx))
    fs_3d_path = os.path.join(result_save_path, 'facial_shape_3d.npy')
    EyeballCenter_path = os.path.join(result_save_path, 'eyeball_center.npy')

    saved_fs_3d = np.load(fs_3d_path)
    saved_EyeballCenter_W = np.load(EyeballCenter_path)

    gaze_I_seq = [[], []]  # [[x_seq], [y_seq]] for kalman filter
    dist_seq = []

    # Start
    cap = cv2.VideoCapture(0)  # chose camera index (try 1, 2, 3)
    cap.set(3, 1280)
    cap.set(4, 720)

    while cap.isOpened():
        success, frame = cap.read()
        if not success:  # no frame input
            continue

        time_0 = time.time()
        image = frame
        gaze_C, gaze_W, extend_dict, exist_face = \
            gaze_vector(image, intrinsic_params, cfg, saved_fs_3d=saved_fs_3d, saved_EyeballCenter_W=saved_EyeballCenter_W)

        # Face 3D model
        if cfg.render:
            image_render = image.copy()
            R_lst, offset_lst, roi_box_lst, size, exist_shape = get_render_params(image_render, tddfa, face_boxes)

            if len(R_lst) != 0:
                R, offset, roi_box = R_lst[0], offset_lst[0], roi_box_lst[0]
                pts3d = R @ saved_fs_3d.T + offset
                pts3d = similar_transform(pts3d, roi_box, size)

                image_render = render(image_render, [pts3d], tddfa.tri, alpha=0.6, show_flag=False, with_bg_flag=False)
            else:
                image_render = np.zeros_like(image_render)

        if exist_face:
            print(gaze_C)
            gaze_I = extend_dict['gaze_I']
            gaze_I[1] = gaze_I[1] - 0.3  # compensation of head pose

            # kalman filter
            if len(gaze_I_seq[0]) >= 3:
                gaze_I_seq[0] = gaze_I_seq[0][1:]
                gaze_I_seq[1] = gaze_I_seq[1][1:]
            gaze_I_seq[0].append(gaze_I[0])
            gaze_I_seq[1].append(gaze_I[1])
            gaze_I_kf = np.array([kalman_filter(gaze_I_seq[0])[-1], kalman_filter(gaze_I_seq[1])[-1]])
            gaze_I = gaze_I_kf

            # params for drawing gaze lines
            k = 45  # length 25
            line_width = 7  # width 3
            color = (0, 255, 255)

            LeftPupil = extend_dict['Pupil_I'][0]
            RightPupil = extend_dict['Pupil_I'][1]

            # change line params when distance get changed for better effects
            dist_to_cam = extend_dict['distance_to_camera']

            # kalman filter
            if len(dist_seq) >= 3:
                dist_seq = dist_seq[1:]
            dist_seq.append(dist_to_cam)
            dist_kf = kalman_filter(dist_seq)[-1]
            dist_to_cam = dist_kf

            k = k * 8e5 / dist_to_cam
            line_width = int(line_width * 8e5 / dist_to_cam)

            LeftEnd = LeftPupil + gaze_I * k
            RightEnd = RightPupil + gaze_I * k

            cv2.arrowedLine(image, LeftPupil.astype(int), LeftEnd.astype(int), color, line_width, 1, 0, 0.08)
            cv2.arrowedLine(image, RightPupil.astype(int), RightEnd.astype(int), color, line_width, 1, 0, 0.08)

            if cfg.render:
                color_render = (0, 0, 255)
                cv2.arrowedLine(image_render, LeftPupil.astype(int), LeftEnd.astype(int), color_render, line_width, 1, 0, 0.08)
                cv2.arrowedLine(image_render, RightPupil.astype(int), RightEnd.astype(int), color_render, line_width, 1, 0, 0.08)

        image = cv2.flip(image, 1)

        # fps
        time_1 = time.time()
        t = time_1 - time_0
        cv2.putText(image, "FPS {0}".format(str(1 / (t + 1e-10))), (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 1)

        cv2.imshow('image', image)

        if cfg.render:
            image_render = cv2.flip(image_render, 1)
            cv2.imshow('render', image_render)

        if cv2.waitKey(1) & 0xFF == 27:
            break

    cap.release()
    cv2.destroyAllWindows()
















