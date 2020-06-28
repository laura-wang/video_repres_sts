import cv2
import numpy as np

from sts.compute_appearance_statistics import compute_app_pattern_1, compute_app_pattern_2, compute_app_pattern_3, compute_app_global


def app_statistics(app_flag, sample):

    app_1, app_2, app_3, app_global = app_flag

    app_label = []
    rgb_clip = sample['rgb_clip']

    trans_rgb_clip = np.transpose(rgb_clip, (1, 2, 3, 0))


    if app_1:
        div_idx_1, div_col_1, con_idx_1, con_col_1 = compute_app_pattern_1(trans_rgb_clip)
        app_label.append(div_idx_1)
        app_label.append(div_col_1)
        app_label.append(con_idx_1)
        app_label.append(con_col_1)



    if app_2:
        div_idx_2, div_col_2, con_idx_2, con_col_2 = compute_app_pattern_2(trans_rgb_clip)
        app_label.append(div_idx_2)
        app_label.append(div_col_2)
        app_label.append(con_idx_2)
        app_label.append(con_col_2)


    if app_3:
        div_idx_3, div_col_3, con_idx_3, con_col_3 = compute_app_pattern_3(trans_rgb_clip)
        app_label.append(div_idx_3)
        app_label.append(div_col_3)
        app_label.append(con_idx_3)
        app_label.append(con_col_3)


    ## global statistics
    if app_global:
        global_domi_color= compute_app_global(trans_rgb_clip)
        app_label.append(global_domi_color)



    sample['app_label'] = np.array(app_label)

    return sample


