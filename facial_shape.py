import numpy as np


def recon_facial_shape(image, cfg):
    exist_fs = False
    fs_3d = np.zeros((1, 1))
    fs_frontal_3d = np.zeros((1, 1))

    if cfg.face_3d_model == '3DDFA_V2':
        from TDDFA_V2.facial_shape import shape
        shape_lst, exist_fs = shape(image)
        fs_3d = shape_lst[0].T  # (38365, 3)

        fs_frontal_3d = fs_3d[: cfg.face_3d_model_frontal_idx, :]

    return fs_3d, fs_frontal_3d, exist_fs



