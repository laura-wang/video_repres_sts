import cv2
import numpy as np
from scipy import ndimage
from sts.compute_motion_statistics_fast import compute_motion_pattern_1, compute_motion_pattern_2, compute_motion_pattern_3, compute_motion_global



def motion_statistics(motion_flag, sample):

    motion_1, motion_2, motion_3, motion_global = motion_flag

    du_x_all, du_y_all, du_x_sum, du_y_sum = compute_motion_boudary(sample['u_flow'])
    dv_x_all, dv_y_all, dv_x_sum, dv_y_sum = compute_motion_boudary(sample['v_flow'])

    motion_label = []
    mag_u, ang_u = cv2.cartToPolar(du_x_sum, du_y_sum, angleInDegrees=True)
    mag_v, ang_v = cv2.cartToPolar(dv_x_sum, dv_y_sum, angleInDegrees=True)

    if motion_1:
        # print('motion_1')
        u_max_mag_1, u_max_ang_1 = compute_motion_pattern_1(mag_u, ang_u)
        v_max_mag_1, v_max_ang_1 = compute_motion_pattern_1(mag_v, ang_v)
        motion_label.append(u_max_mag_1)
        motion_label.append(u_max_ang_1)
        motion_label.append(v_max_mag_1)
        motion_label.append(v_max_ang_1)

    if motion_2:
        # print('motion_2')
        u_max_mag_2, u_max_ang_2 = compute_motion_pattern_2(mag_u, ang_u)
        v_max_mag_2, v_max_ang_2 = compute_motion_pattern_2(mag_v, ang_v)
        motion_label.append(u_max_mag_2)
        motion_label.append(u_max_ang_2)
        motion_label.append(v_max_mag_2)
        motion_label.append(v_max_ang_2)

    if motion_3:
        # print('motion_3')
        u_max_mag_3, u_max_ang_3 = compute_motion_pattern_3(mag_u, ang_u)
        v_max_mag_3, v_max_ang_3 = compute_motion_pattern_3(mag_v, ang_v)

        motion_label.append(u_max_mag_3)
        motion_label.append(u_max_ang_3)
        motion_label.append(v_max_mag_3)
        motion_label.append(v_max_ang_3)

    ## global statistics
    if motion_global:
        # print('motion_global')
        max_du_idx = compute_motion_global(du_x_all, du_y_all)
        max_dv_idx = compute_motion_global(dv_x_all, dv_y_all)

        motion_label.append(max_du_idx)
        motion_label.append(max_dv_idx)

    sample['motion_label'] = np.array(motion_label)

    # for debug
    sample['du'] = {'du_x_all': du_x_all, 'du_y_all': du_y_all, 'du_x_sum': du_x_sum, 'du_y_sum': du_y_sum}
    sample['dv'] = {'dv_x_all': dv_x_all, 'dv_y_all': dv_y_all, 'dv_x_sum': dv_x_sum, 'dv_y_sum': dv_y_sum}

    return sample

def compute_motion_boudary(flow_clip):

    mx = np.array([[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]])
    my = np.array([[-1, -1, -1], [0, 0, 0], [1, 1, 1]])

    dx_all = []
    dy_all = []

    mb_x = 0
    mb_y = 0


    for flow_img in flow_clip:
        d_x = ndimage.convolve(flow_img, mx)
        d_y = ndimage.convolve(flow_img, my)

        dx_all.append(d_x)
        dy_all.append(d_y)

        mb_x += d_x
        mb_y += d_y

    dx_all = np.array(dx_all)
    dy_all = np.array(dy_all)

    return dx_all, dy_all, mb_x, mb_y