import cv2
import time
import numpy as np
from scipy import io


def relative(landmark, shape):
    """
    :param landmark:
    :param shape: h * w * 3
    :return:
    """
    return np.array([int(landmark.x * shape[1]), int(landmark.y * shape[0])])


def head_pose(lmk_2d, lmk_3d, intrinsic_params):
    camera_matrix, dist_coeffs = intrinsic_params['camera_matrix'], intrinsic_params['dist_coeffs']
    success, rotation_vector, translation_vector = cv2.solvePnP(lmk_3d, lmk_2d,
                                                                camera_matrix, dist_coeffs,
                                                                flags=cv2.SOLVEPNP_ITERATIVE)
    rotation_matrix = cv2.Rodrigues(rotation_vector)[0]
    return rotation_vector, rotation_matrix, translation_vector


def W_2_I(vers_W, rotation_vector, translation_vector, intrinsic_params):
    """
    :param vers_W: shape = n * 3
    :param rotation_vector:
    :param translation_vector:
    :param intrinsic_params:
    :return: shape = n * 2, int
    """
    vers_I, _ = cv2.projectPoints(vers_W, rotation_vector, translation_vector, intrinsic_params['camera_matrix'], intrinsic_params['dist_coeffs'])
    return vers_I.squeeze().astype(int)


def C_2_I(vers_C, intrinsic_params):
    if len(vers_C.shape) == 1:
        vers_C = np.reshape(vers_C, (1, vers_C.shape[0]))

    vers_C = vers_C.T
    vers_C = np.matmul(intrinsic_params['camera_matrix'], vers_C)
    vers_C = vers_C.T

    return vers_C.squeeze()


def I_2_W(vers_I, rotation_matrix, translation_vector, intrinsic_params, scalar=1):  # (n, 2)
    if len(vers_I.shape) == 1:
        vers_I = np.reshape(vers_I, (1, vers_I.shape[0]))

    vers_I_homo = np.column_stack((vers_I, np.ones((vers_I.shape[0], 1)))) * scalar
    vers_I_homo = vers_I_homo.T

    camera_matrix = intrinsic_params['camera_matrix']
    vers_W = np.matmul(np.linalg.inv(rotation_matrix),
                       (np.matmul(np.linalg.inv(camera_matrix), vers_I_homo) - translation_vector)).T
    return vers_W.squeeze()


def W_2_C(vers_W, rotation_matrix, translation_vector):
    """
    :param vers_W: shape = n * 3 or (, 3)
    :param rotation_matrix:
    :param translation_vector:
    :return: shape = n * 3 or (, 3)
    """
    if len(vers_W.shape) == 1:
        vers_W = np.reshape(vers_W, (1, vers_W.shape[0]))

    vers_W = vers_W.T
    vers_C = np.matmul(rotation_matrix, vers_W) + translation_vector
    vers_C = vers_C.T

    if (vers_C[:, 2] < 0).all():
        vers_C = - vers_C

    return vers_C.squeeze()


def eye_mask(Eye_contour_2d, fs_frontal_I, fs_frontal_W, shape, cfg):
    if cfg.eye_mask_type == 'simple':
        eye_mask_dict = io.loadmat(cfg.eye_mask_3d_idx_path)
        LeftEye_mask_idx = eye_mask_dict['LeftEye_mask']
        RightEye_mask_idx = eye_mask_dict['RightEye_mask']

        LeftEye_mask_W = fs_frontal_W[LeftEye_mask_idx[0][0]: LeftEye_mask_idx[0][1], :]
        for i in range(1, LeftEye_mask_idx.shape[0]):
            idx_1, idx_2 = LeftEye_mask_idx[i]
            rows = fs_frontal_W[idx_1: idx_2, :]
            LeftEye_mask_W = np.vstack((LeftEye_mask_W, rows))

        RightEye_mask_W = fs_frontal_W[RightEye_mask_idx[0][0]: RightEye_mask_idx[0][1], :]
        for i in range(1, RightEye_mask_idx.shape[0]):
            idx_1, idx_2 = RightEye_mask_idx[i]
            rows = fs_frontal_W[idx_1: idx_2, :]
            RightEye_mask_W = np.vstack((RightEye_mask_W, rows))

        return LeftEye_mask_W, RightEye_mask_W

    if cfg.eye_mask_type == 'contour':
        left_zeros = np.zeros(shape, dtype=np.uint8)
        right_zeros = np.zeros(shape, dtype=np.uint8)
        LeftEye_mask_I = cv2.fillPoly(left_zeros, [Eye_contour_2d[0]], (255, 255, 255))
        RightEye_mask_I = cv2.fillPoly(right_zeros, [Eye_contour_2d[1]], (255, 255, 255))

        LeftEye_mask_I = cv2.cvtColor(LeftEye_mask_I, cv2.COLOR_BGR2GRAY).astype(int)  # 255
        RightEye_mask_I = cv2.cvtColor(RightEye_mask_I, cv2.COLOR_BGR2GRAY).astype(int)

        # Speed up to 0.003s
        LeftEye_region = \
            (fs_frontal_I[:, 0] <= np.max(Eye_contour_2d[0, :, 0])) & (fs_frontal_I[:, 0] >= np.min(Eye_contour_2d[0, :, 0])) &\
            (fs_frontal_I[:, 1] <= np.max(Eye_contour_2d[0, :, 1])) & (fs_frontal_I[:, 1] >= np.min(Eye_contour_2d[0, :, 1]))
        RightEye_region = \
            (fs_frontal_I[:, 0] <= np.max(Eye_contour_2d[1, :, 0])) & (fs_frontal_I[:, 0] >= np.min(Eye_contour_2d[1, :, 0])) & \
            (fs_frontal_I[:, 1] <= np.max(Eye_contour_2d[1, :, 1])) & (fs_frontal_I[:, 1] >= np.min(Eye_contour_2d[1, :, 1]))

        fs_LeftEye_region_I = fs_frontal_I[LeftEye_region]
        fs_LeftEye_region_W = fs_frontal_W[LeftEye_region]
        fs_RightEye_region_I = fs_frontal_I[RightEye_region]
        fs_RightEye_region_W = fs_frontal_W[RightEye_region]

        mm_left = LeftEye_mask_I[fs_LeftEye_region_I[:, 1], :][:, fs_LeftEye_region_I[:, 0]]  # only diagonal useful
        mm_left = mm_left.diagonal()
        LeftEye_mask_W = fs_LeftEye_region_W[np.where(mm_left == 255)]

        mm_right = RightEye_mask_I[fs_RightEye_region_I[:, 1], :][:, fs_RightEye_region_I[:, 0]]  # only diagonal useful
        mm_right = mm_right.diagonal()
        RightEye_mask_W = fs_RightEye_region_W[np.where(mm_right == 255)]

        return LeftEye_mask_W, RightEye_mask_W


def pupil_center(Pupil_2d, LeftEye_mask_W, RightEye_mask_W,
                 rotation_vector, translation_vector, intrinsic_params):
    # Speed up
    LeftEye_mask_I, _ = cv2.projectPoints(
        LeftEye_mask_W, rotation_vector, translation_vector, intrinsic_params['camera_matrix'], intrinsic_params['dist_coeffs'])
    RightEye_mask_I, _ = cv2.projectPoints(
        RightEye_mask_W, rotation_vector, translation_vector, intrinsic_params['camera_matrix'], intrinsic_params['dist_coeffs'])

    LeftEye_mask_I = LeftEye_mask_I.squeeze(1)
    RightEye_mask_I = RightEye_mask_I.squeeze(1)

    LeftPupil_idx = np.argmin(np.linalg.norm(LeftEye_mask_I - np.array(Pupil_2d[0]).astype(float), axis=1))
    LeftPupil_W = LeftEye_mask_W[LeftPupil_idx]

    RightPupil_idx = np.argmin(np.linalg.norm(RightEye_mask_I - np.array(Pupil_2d[1]).astype(float), axis=1))
    RightPupil_W = RightEye_mask_W[RightPupil_idx]

    return LeftPupil_W, RightPupil_W


def eyeball_center(LeftEye_mask_W, RightEye_mask_W):
    LeftEyeballCenter_W, left_ra = spherical_fitting(LeftEye_mask_W)
    RightEyeballCenter_W, right_ra = spherical_fitting(RightEye_mask_W)
    return LeftEyeballCenter_W, RightEyeballCenter_W


def average_gaze(LeftPupil, RightPupil, LeftEyeballCenter, RightEyeballCenter):  # W or C

    left_gaze = LeftPupil - LeftEyeballCenter
    right_gaze = RightPupil - RightEyeballCenter
    left_gaze_norm = left_gaze / (np.linalg.norm(left_gaze) + 1e-10)
    right_gaze_norm = right_gaze / (np.linalg.norm(right_gaze) + 1e-10)

    avg_gaze = (left_gaze_norm + right_gaze_norm) / 2
    return avg_gaze


def spherical_fitting(points):  # shape = n * 3
    """https://www.2bboy.com/archives/171.html"""
    points = points.astype(float)
    num_points = points.shape[0]
    # print(num_points)
    x = points[:, 0]
    y = points[:, 1]
    z = points[:, 2]
    x_avr = sum(x) / num_points
    y_avr = sum(y) / num_points
    z_avr = sum(z) / num_points
    xx_avr = sum(x * x) / num_points
    yy_avr = sum(y * y) / num_points
    zz_avr = sum(z * z) / num_points
    xy_avr = sum(x * y) / num_points
    xz_avr = sum(x * z) / num_points
    yz_avr = sum(y * z) / num_points
    xxx_avr = sum(x * x * x) / num_points
    xxy_avr = sum(x * x * y) / num_points
    xxz_avr = sum(x * x * z) / num_points
    xyy_avr = sum(x * y * y) / num_points
    xzz_avr = sum(x * z * z) / num_points
    yyy_avr = sum(y * y * y) / num_points
    yyz_avr = sum(y * y * z) / num_points
    yzz_avr = sum(y * z * z) / num_points
    zzz_avr = sum(z * z * z) / num_points

    A = np.array([[xx_avr - x_avr * x_avr, xy_avr - x_avr * y_avr, xz_avr - x_avr * z_avr],
                  [xy_avr - x_avr * y_avr, yy_avr - y_avr * y_avr, yz_avr - y_avr * z_avr],
                  [xz_avr - x_avr * z_avr, yz_avr - y_avr * z_avr, zz_avr - z_avr * z_avr]])
    b = np.array([xxx_avr - x_avr * xx_avr + xyy_avr - x_avr * yy_avr + xzz_avr - x_avr * zz_avr,
                  xxy_avr - y_avr * xx_avr + yyy_avr - y_avr * yy_avr + yzz_avr - y_avr * zz_avr,
                  xxz_avr - z_avr * xx_avr + yyz_avr - z_avr * yy_avr + zzz_avr - z_avr * zz_avr])

    b = b / 2
    center = np.linalg.solve(A, b)
    x0 = center[0]
    y0 = center[1]
    z0 = center[2]
    r2 = xx_avr - 2 * x0 * x_avr + x0 * x0 + yy_avr - 2 * y0 * y_avr + y0 * y0 + zz_avr - 2 * z0 * z_avr + z0 * z0
    r = r2 ** 0.5
    return center, r


def lines_intersection(PList, LineD):
    l = len(LineD)
    I = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
    q = np.array([0, 0, 0])
    Msum = np.array([[0, 0, 0], [0, 0, 0], [0, 0, 0]])

    for j in range(l):
        Lnormal = np.array([-LineD[j] / (np.linalg.norm(LineD[j]) + 1e-10)])
        p = PList[j][:3]
        viviT = Lnormal * Lnormal.T
        M = I - viviT
        Msum = Msum + M
        q = q + np.dot(M, p)
    q = q.T
    Msum_inv = np.linalg.inv(Msum)
    return Msum_inv @ q.T


def get_distance_point2line(point, line_point1, line_point2):  # (, 3)
    vec1 = line_point1 - point
    vec2 = line_point2 - point
    distance = np.linalg.norm(np.cross(vec1, vec2)) / np.linalg.norm(line_point1 - line_point2)
    return distance


def angular(gaze, label):  # angle between two vectors
    total = np.sum(gaze * label)
    return np.arccos(min(total / (np.linalg.norm(gaze) * np.linalg.norm(label)), 0.9999999)) * 180 / np.pi


def kalman_filter(data, q=0.0001, r=0.01):
    x0 = data[0]
    p0 = 1.0

    x = [x0]
    for z in data[1:]:
        x1_minus = x0                         # X(k|k-1) = AX(k-1|k-1) + BU(k) + W(k), A=1,BU(k) = 0
        p1_minus = p0 + q                     # P(k|k-1) = AP(k-1|k-1)A' + Q(k), A=1

        k1 = p1_minus / (p1_minus + r)        # Kg(k)=P(k|k-1)H'/[HP(k|k-1)H' + R], H=1
        x0 = x1_minus + k1 * (z - x1_minus)   # X(k|k) = X(k|k-1) + Kg(k)[Z(k) - HX(k|k-1)], H=1
        p0 = (1 - k1) * p1_minus              # P(k|k) = (1 - Kg(k)H)P(k|k-1), H=1
        x.append(x0)
    return x








