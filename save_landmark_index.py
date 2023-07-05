import pickle
import numpy as np
from scipy import io


lmk_idx_dict = {
        'LeftEye_left':   {'lmk_idx': 0,  '2d_idx': 263, '3d_idx': 13551},
        'LeftEye_right':  {'lmk_idx': 1,  '2d_idx': 382, '3d_idx': 10327},
        'RightEye_left':  {'lmk_idx': 2,  '2d_idx': 133, '3d_idx': 6086},
        'RightEye_right': {'lmk_idx': 3,  '2d_idx': 33,  '3d_idx': 2474},
        'Nose':           {'lmk_idx': 4,  '2d_idx': 4,   '3d_idx': 8191},
        'Mouse_left':     {'lmk_idx': 5,  '2d_idx': 292, '3d_idx': 10923},
        'Mouse_right':    {'lmk_idx': 6,  '2d_idx': 62,  '3d_idx': 5520},
        'Lip_top':        {'lmk_idx': 7,  '2d_idx': 0,   '3d_idx': 8215},
        'Chin':           {'lmk_idx': 8,  '2d_idx': 175, '3d_idx': 36086},
}

with open("landmark_index.pkl", "wb") as lmk_idx_file:
    pickle.dump(lmk_idx_dict, lmk_idx_file)


eye_mask_dict = {
        'LeftEye_mask': np.array([
                (10455, 10458),
                (10584, 10588),
                (10713, 10718),
                (10842, 10848),
                (10970, 10978),
                (11098, 11107),
                (11226, 11236),
                (11354, 11365),
                (11482, 11494),
                (11611, 11623),
                (11740, 11752),
                (11869, 11881),
                (11998, 12010),
                (12127, 12139),
                (12256, 12268),
                (12385, 12397),
                (12514, 12526),
                (12643, 12655),
                (12772, 12784),
                (12902, 12912),
                (13032, 13041),
                (13162, 13168),
                (13291, 13297),
                (13421, 13425),
                (13550, 13553)]),

        'RightEye_mask': np.array([
                (2732, 2734),
                (2861, 2863),
                (2989, 2992),
                (3118, 3122),
                (3246, 3252),
                (3374, 3382),
                (3502, 3511),
                (3631, 3641),
                (3760, 3771),
                (3889, 3900),
                (4018, 4029),
                (4147, 4158),
                (4276, 4287),
                (4405, 4416),
                (4534, 4545),
                (4663, 4674),
                (4792, 4803),
                (4922, 4932),
                (5051, 5061),
                (5181, 5190),
                (5311, 5318),
                (5441, 5446),
                (5570, 5574),
                (5699, 5702)])
}
io.savemat('eye_mask_3d_index.mat', mdict=eye_mask_dict)


eye_contour_dict = {
        'LeftEye_mask': np.array(
                [362, 398, 384, 385, 386, 387, 388, 466,
                 263, 249, 390, 373, 374, 380, 381, 382]),
        'RightEye_mask': np.array(
                [33, 7, 163, 144, 145, 153, 154, 155,
                 133, 173, 157, 158, 159, 160, 161, 246])
}
io.savemat('eye_contour_2d_index.mat', mdict=eye_contour_dict)









