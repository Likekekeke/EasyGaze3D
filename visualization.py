import os
import sys
import cv2
import pickle
import numpy as np
import mediapipe as mp
import matplotlib.pyplot as plt


def visualize_3d(vis_dict):
    fig = plt.figure(figsize=(20, 20))
    ax = fig.add_subplot(111, projection='3d')
    ax.view_init(vis_dict['view'][0], vis_dict['view'][1])

    vis_keys = vis_dict.keys()
    xmin, xmax, ymin, ymax, zmin, zmax = 1e10, -1e10, 1e10, -1e10, 1e10, -1e10

    if 'facial_shape_3d' in vis_keys:
        fs_3d = vis_dict['facial_shape_3d']['obj']
        fs_3d = fs_3d[:: 5, :]

        xmin = np.min([xmin, np.min(fs_3d[:, 0])])
        xmax = np.max([xmax, np.max(fs_3d[:, 0])])
        ymin = np.min([ymin, np.min(fs_3d[:, 1])])
        ymax = np.max([ymax, np.max(fs_3d[:, 1])])
        zmin = np.min([zmin, np.min(fs_3d[:, 2])])
        zmax = np.max([zmax, np.max(fs_3d[:, 2])])

        x_fs_3d = fs_3d[:, 0]
        y_fs_3d = fs_3d[:, 1]
        z_fs_3d = fs_3d[:, 2]
        ax.scatter(x_fs_3d, y_fs_3d, z_fs_3d, c=z_fs_3d, cmap='rainbow',
                   s=vis_dict['facial_shape_3d']['s'], alpha=vis_dict['facial_shape_3d']['alpha'])

    if 'LeftPupil' in vis_keys:
        LeftPupil = vis_dict['LeftPupil']['obj']
        LeftPupil = np.reshape(LeftPupil, (1, 3))

        xmin = np.min([xmin, np.min(LeftPupil[:, 0])])
        xmax = np.max([xmax, np.max(LeftPupil[:, 0])])
        ymin = np.min([ymin, np.min(LeftPupil[:, 1])])
        ymax = np.max([ymax, np.max(LeftPupil[:, 1])])
        zmin = np.min([zmin, np.min(LeftPupil[:, 2])])
        zmax = np.max([zmax, np.max(LeftPupil[:, 2])])

        x_LeftPupil = LeftPupil[:, 0]
        y_LeftPupil = LeftPupil[:, 1]
        z_LeftPupil = LeftPupil[:, 2]
        ax.scatter(x_LeftPupil, y_LeftPupil, z_LeftPupil,
                   c=vis_dict['LeftPupil']['color'], s=vis_dict['LeftPupil']['s'], alpha=vis_dict['LeftPupil']['alpha'])

    if 'RightPupil' in vis_keys:
        LeftPupil = vis_dict['RightPupil']['obj']
        LeftPupil = np.reshape(LeftPupil, (1, 3))

        xmin = np.min([xmin, np.min(LeftPupil[:, 0])])
        xmax = np.max([xmax, np.max(LeftPupil[:, 0])])
        ymin = np.min([ymin, np.min(LeftPupil[:, 1])])
        ymax = np.max([ymax, np.max(LeftPupil[:, 1])])
        zmin = np.min([zmin, np.min(LeftPupil[:, 2])])
        zmax = np.max([zmax, np.max(LeftPupil[:, 2])])

        x_LeftPupil = LeftPupil[:, 0]
        y_LeftPupil = LeftPupil[:, 1]
        z_LeftPupil = LeftPupil[:, 2]
        ax.scatter(x_LeftPupil, y_LeftPupil, z_LeftPupil,
                   c=vis_dict['RightPupil']['color'], s=vis_dict['RightPupil']['s'], alpha=vis_dict['RightPupil']['alpha'])

    if 'LeftEye_mask' in vis_keys:
        fs_3d = vis_dict['LeftEye_mask']['obj']

        xmin = np.min([xmin, np.min(fs_3d[:, 0])])
        xmax = np.max([xmax, np.max(fs_3d[:, 0])])
        ymin = np.min([ymin, np.min(fs_3d[:, 1])])
        ymax = np.max([ymax, np.max(fs_3d[:, 1])])
        zmin = np.min([zmin, np.min(fs_3d[:, 2])])
        zmax = np.max([zmax, np.max(fs_3d[:, 2])])

        x_fs_3d = fs_3d[:, 0]
        y_fs_3d = fs_3d[:, 1]
        z_fs_3d = fs_3d[:, 2]
        ax.scatter(x_fs_3d, y_fs_3d, z_fs_3d, c=z_fs_3d, cmap='rainbow',
                   s=vis_dict['LeftEye_mask']['s'], alpha=vis_dict['LeftEye_mask']['alpha'])

    if 'RightEye_mask' in vis_keys:
        fs_3d = vis_dict['RightEye_mask']['obj']

        xmin = np.min([xmin, np.min(fs_3d[:, 0])])
        xmax = np.max([xmax, np.max(fs_3d[:, 0])])
        ymin = np.min([ymin, np.min(fs_3d[:, 1])])
        ymax = np.max([ymax, np.max(fs_3d[:, 1])])
        zmin = np.min([zmin, np.min(fs_3d[:, 2])])
        zmax = np.max([zmax, np.max(fs_3d[:, 2])])

        x_fs_3d = fs_3d[:, 0]
        y_fs_3d = fs_3d[:, 1]
        z_fs_3d = fs_3d[:, 2]
        ax.scatter(x_fs_3d, y_fs_3d, z_fs_3d, c=z_fs_3d, cmap='rainbow',
                   s=vis_dict['RightEye_mask']['s'], alpha=vis_dict['RightEye_mask']['alpha'])

    if 'landmark' in vis_keys:
        LeftPupil = vis_dict['landmark']['obj']

        xmin = np.min([xmin, np.min(LeftPupil[:, 0])])
        xmax = np.max([xmax, np.max(LeftPupil[:, 0])])
        ymin = np.min([ymin, np.min(LeftPupil[:, 1])])
        ymax = np.max([ymax, np.max(LeftPupil[:, 1])])
        zmin = np.min([zmin, np.min(LeftPupil[:, 2])])
        zmax = np.max([zmax, np.max(LeftPupil[:, 2])])

        x_LeftPupil = LeftPupil[:, 0]
        y_LeftPupil = LeftPupil[:, 1]
        z_LeftPupil = LeftPupil[:, 2]
        ax.scatter(x_LeftPupil, y_LeftPupil, z_LeftPupil,
                   c=vis_dict['landmark']['color'], s=vis_dict['landmark']['s'], alpha=vis_dict['landmark']['alpha'])

    if 'LeftEyeballCenter' in vis_keys:
        LeftPupil = vis_dict['LeftEyeballCenter']['obj']
        LeftPupil = np.reshape(LeftPupil, (1, 3))

        xmin = np.min([xmin, np.min(LeftPupil[:, 0])])
        xmax = np.max([xmax, np.max(LeftPupil[:, 0])])
        ymin = np.min([ymin, np.min(LeftPupil[:, 1])])
        ymax = np.max([ymax, np.max(LeftPupil[:, 1])])
        zmin = np.min([zmin, np.min(LeftPupil[:, 2])])
        zmax = np.max([zmax, np.max(LeftPupil[:, 2])])

        x_LeftPupil = LeftPupil[:, 0]
        y_LeftPupil = LeftPupil[:, 1]
        z_LeftPupil = LeftPupil[:, 2]
        ax.scatter(x_LeftPupil, y_LeftPupil, z_LeftPupil,
                   c=vis_dict['LeftEyeballCenter']['color'], s=vis_dict['LeftEyeballCenter']['s'], alpha=vis_dict['LeftEyeballCenter']['alpha'])

    if 'RightEyeballCenter' in vis_keys:
        LeftPupil = vis_dict['RightEyeballCenter']['obj']
        LeftPupil = np.reshape(LeftPupil, (1, 3))

        xmin = np.min([xmin, np.min(LeftPupil[:, 0])])
        xmax = np.max([xmax, np.max(LeftPupil[:, 0])])
        ymin = np.min([ymin, np.min(LeftPupil[:, 1])])
        ymax = np.max([ymax, np.max(LeftPupil[:, 1])])
        zmin = np.min([zmin, np.min(LeftPupil[:, 2])])
        zmax = np.max([zmax, np.max(LeftPupil[:, 2])])

        x_LeftPupil = LeftPupil[:, 0]
        y_LeftPupil = LeftPupil[:, 1]
        z_LeftPupil = LeftPupil[:, 2]
        ax.scatter(x_LeftPupil, y_LeftPupil, z_LeftPupil,
                   c=vis_dict['RightEyeballCenter']['color'], s=vis_dict['RightEyeballCenter']['s'], alpha=vis_dict['RightEyeballCenter']['alpha'])

    if 'gaze' in vis_keys:
        gaze_vector = vis_dict['gaze']['obj']
        LeftPupil = vis_dict['gaze']['LeftPupil']
        RightPupil = vis_dict['gaze']['RightPupil']

        left_end = LeftPupil + gaze_vector * vis_dict['gaze']['scalar']
        right_end = RightPupil + gaze_vector * vis_dict['gaze']['scalar']

        plt.plot([LeftPupil[0], left_end[0]],
                 [LeftPupil[1], left_end[1]],
                 [LeftPupil[2], left_end[2]],
                 c=vis_dict['gaze']['color'], linewidth=vis_dict['gaze']['linewidth'], alpha=vis_dict['gaze']['alpha'])
        plt.plot([RightPupil[0], right_end[0]],
                 [RightPupil[1], right_end[1]],
                 [RightPupil[2], right_end[2]],
                 c=vis_dict['gaze']['color'], linewidth=vis_dict['gaze']['linewidth'], alpha=vis_dict['gaze']['alpha'])

    if 'gaze_lines_dict' in vis_keys:
        left_lines_strt = vis_dict['gaze_lines_dict']['obj']['left_lines_strt']
        left_lines_vec = vis_dict['gaze_lines_dict']['obj']['left_lines_vec']
        right_lines_strt = vis_dict['gaze_lines_dict']['obj']['right_lines_strt']
        right_lines_vec = vis_dict['gaze_lines_dict']['obj']['right_lines_vec']

        left_lines_strt = left_lines_strt[:: int(1 / vis_dict['gaze_lines_dict']['sampling_rate']), :]
        left_lines_vec = left_lines_vec[:: int(1 / vis_dict['gaze_lines_dict']['sampling_rate']), :]

        left_lines_strt = left_lines_strt - vis_dict['gaze_lines_dict']['start_pos'] * left_lines_vec
        left_lines_end = left_lines_strt + vis_dict['gaze_lines_dict']['end_pos'] * left_lines_vec

        for i in range(left_lines_strt.shape[0]):
            plt.plot([left_lines_strt[i][0], left_lines_end[i][0]],
                     [left_lines_strt[i][1], left_lines_end[i][1]],
                     [left_lines_strt[i][2], left_lines_end[i][2]],
                     c=vis_dict['gaze_lines_dict']['left_color'], linewidth=vis_dict['gaze_lines_dict']['linewidth'], alpha=vis_dict['gaze_lines_dict']['alpha'])

        right_lines_strt = right_lines_strt[:: int(1 / vis_dict['gaze_lines_dict']['sampling_rate']), :]
        right_lines_vec = right_lines_vec[:: int(1 / vis_dict['gaze_lines_dict']['sampling_rate']), :]

        right_lines_strt = right_lines_strt - vis_dict['gaze_lines_dict']['start_pos'] * right_lines_vec
        right_lines_end = right_lines_strt + vis_dict['gaze_lines_dict']['end_pos'] * right_lines_vec

        for i in range(left_lines_strt.shape[0]):
            plt.plot([right_lines_strt[i][0], right_lines_end[i][0]],
                     [right_lines_strt[i][1], right_lines_end[i][1]],
                     [right_lines_strt[i][2], right_lines_end[i][2]],
                     c=vis_dict['gaze_lines_dict']['right_color'], linewidth=vis_dict['gaze_lines_dict']['linewidth'], alpha=vis_dict['gaze_lines_dict']['alpha'])

    x_center = (xmin + xmax) / 2
    y_center = (ymin + ymax) / 2
    z_center = (zmin + zmax) / 2
    x_range = xmax - xmin
    y_range = ymax - ymin
    z_range = zmax - zmin
    max_range = np.max([x_range, y_range, z_range]) * 1.2

    xmin = x_center - max_range / 2
    xmax = x_center + max_range / 2
    ymin = y_center - max_range / 2
    ymax = y_center + max_range / 2
    zmin = z_center - max_range / 2
    zmax = z_center + max_range / 2

    if vis_dict['CS'] == 'C':
        ax.set_xlim(xmax, xmin)
        ax.set_ylim(ymin, ymax)
        ax.set_zlim(zmax, zmin)
    elif vis_dict['CS'] == 'W':
        ax.set_xlim(xmin, xmax)
        ax.set_ylim(ymin, ymax)
        ax.set_zlim(zmin, zmax)

    ax.w_xaxis.line.set_color("lightgray")
    ax.w_yaxis.line.set_color("lightgray")
    ax.w_zaxis.line.set_color("lightgray")

    if vis_dict['show_tick']:
        ax.set_xlabel('x', size=15)
        ax.set_ylabel('y', size=15)
        ax.set_zlabel('z', size=15)
        plt.tick_params(axis='both', which='major', labelsize=8, labelcolor='black')

    if not vis_dict['show_tick']:
        ax.xaxis._axinfo['tick']['outward_factor'] = 0
        ax.xaxis._axinfo['tick']['inward_factor'] = 0
        ax.yaxis._axinfo['tick']['outward_factor'] = 0
        ax.yaxis._axinfo['tick']['inward_factor'] = 0
        ax.zaxis._axinfo['tick']['outward_factor'] = 0
        ax.zaxis._axinfo['tick']['inward_factor'] = 0

        plt.tick_params(axis='both', which='major', labelbottom=False, labelleft=False)

    plt.show()






