import numpy as np
import cv2
import time

def comput_IoU_one(block, channel, pattern):
    _, h, w, _ = block.shape
    hist_volume = []

    for i in range(16):
        cur_block = block[i, ...]
        hist = cv2.calcHist([cur_block], [channel], None, [256], [0, 256])  # cv2.calcHist can only apply on images
        if pattern == 2:
            if h == 56:
                hist[0] = hist[0] - 28 * 28
            if h == 84:
                hist[0] = hist[0] - 56 * 56
            if h == 112:
                hist[0] = hist[0] - 84 * 84
        elif pattern == 3:
            hist[0] = hist[0] - 1540  # 56*56 - (56*57/2)

        hist_volume.append(hist)

    hist_volume = np.array(hist_volume)
    hist_volume = np.reshape(hist_volume, (16, 256))
    overlap = np.min(hist_volume, 0)
    union = np.max(hist_volume, 0)

    IoU = np.sum(overlap) / np.sum(union)

    return IoU

def comput_IoU_all(block, pattern):
    IoU_r = comput_IoU_one(block, 2, pattern)
    IoU_b = comput_IoU_one(block, 1, pattern)
    IoU_g = comput_IoU_one(block, 0, pattern)

    return (IoU_r+IoU_b+IoU_g) / 3.


##　super slow
def compute_dominant_color_traverse(video_block, pattern): #  16 x H x W x C

    bins = [0] * 8

    for frame in video_block:
        h, w, c = frame.shape
        frame = np.reshape(frame, (h * w, 3))
        for i in range(frame.shape[0]):
            cur_pixel = frame[i]  # b,g,r
            r = cur_pixel[2]
            g = cur_pixel[1]
            b = cur_pixel[0]

            if 0 <= r <= 127 and 0 <= g <= 127 and 128 <= b <= 255:
                bins[0] += 1
            elif 0 <= r <= 127 and 128 <= g <= 255 and 128 <= b <= 255:
                bins[1] += 1
            elif 128 <= r <= 255 and 0 <= g <= 127 and 128 <= b <= 255:
                bins[2] += 1
            elif 128 <= r <= 255 and 128 <= g <= 255 and 128 <= b <= 255:
                bins[3] += 1
            elif 0 <= r <= 127 and 0 <= g <= 127 and 0 <= b <= 127:
                bins[4] += 1
            elif 0 <= r <= 127 and 128 <= g <= 255 and 0 <= b <= 127:
                bins[5] += 1
            elif 128 <= r <= 255 and 0 <= g <= 127 and 0 <= b <= 127:
                bins[6] += 1
            elif 128 <= r <= 255 and 128 <= g <= 255 and 0 <= b <= 127:
                bins[7] += 1

        if pattern == 2:
            if h == 56:
                bins[4] = bins[4] - 28 * 28
            if h == 84:
                bins[4] = bins[4] - 56 * 56
            if h == 112:
                bins[4] = bins[4] - 84 * 84
        elif pattern == 3:
            bins[4] = bins[4] - 1540

    domi_color = np.argmax(bins)
    return domi_color+1

## refer to https://stackoverflow.com/questions/50899692/most-dominant-color-in-rgb-image-opencv-numpy-python

def compute_dominant_color_bincount(a): # maybe only faster for larger block
    a2D = a.reshape(-1,a.shape[-1])
    col_range = (256, 256, 256) # generically : a2D.max(0)+1
    a1D = np.ravel_multi_index(a2D.T, col_range)

    b,g,r = np.unravel_index(np.bincount(a1D).argmax(), col_range)

    if 0 <= r <= 127 and 0 <= g <= 127 and 128 <= b <= 255:
        domi_color = 0
    elif 0 <= r <= 127 and 128 <= g <= 255 and 128 <= b <= 255:
        domi_color = 1
    elif 128 <= r <= 255 and 0 <= g <= 127 and 128 <= b <= 255:
        domi_color = 2
    elif 128 <= r <= 255 and 128 <= g <= 255 and 128 <= b <= 255:
        domi_color = 3
    elif 0 <= r <= 127 and 0 <= g <= 127 and 0 <= b <= 127:
        domi_color = 4
    elif 0 <= r <= 127 and 128 <= g <= 255 and 0 <= b <= 127:
        domi_color = 5
    elif 128 <= r <= 255 and 0 <= g <= 127 and 0 <= b <= 127:
        domi_color = 6
    elif 128 <= r <= 255 and 128 <= g <= 255 and 0 <= b <= 127:
        domi_color = 7

    return domi_color

def compute_div_cont_block_idx(block_all, pattern):
    IoU_all = []
    for cur_block in block_all:
        # cur_block = block_all[i]
        cur_IoU = comput_IoU_all(cur_block, pattern)
        IoU_all.append(cur_IoU)

    diverse_block_idx = np.argmin(IoU_all)
    consist_block_idx = np.argmax(IoU_all)

    return diverse_block_idx, consist_block_idx



def compute_block_pattern_3(block, lu=True):
    block_1 = []  # 16 x 56 x 56 x 3
    block_2 = []
    for i in range(16):
        block_1_bgr = []
        block_2_bgr = []
        cur_block = block[i]
        if lu == False:
            cur_block = np.flip(cur_block, 1)

        for j in range(3):
            tmp_block = cur_block[:, :, j]

            if lu:
                tmp_block_1 = np.tril(tmp_block)
                tmp_block_2 = np.triu(tmp_block)
            else:
                tmp_block_1 = np.triu(tmp_block)
                tmp_block_2 = np.tril(tmp_block)

            block_1_bgr.append(tmp_block_1)
            block_2_bgr.append(tmp_block_2)

        block_1_bgr = np.array(block_1_bgr)
        block_2_bgr = np.array(block_2_bgr)

        block_1.append(block_1_bgr)
        block_2.append(block_2_bgr)

    block_1 = np.array(block_1)
    block_1 = np.transpose(block_1, (0, 2, 3, 1))  #  16 x 3 x 56 x 56 --->  16 x 56 x 56 x 3
    block_2 = np.array(block_2)
    block_2 = np.transpose(block_2, (0, 2, 3, 1))  #  16 x 3 x 56 x 56 --->  16 x 56 x 56 x 3

    return block_1, block_2


def compute_pattern_2_color(block_all, idx):

    block = block_all[idx]  # D x H x W x C

    D, H, W, C = block.shape
    if H == 28:
        domi_color = compute_dominant_color_bincount(block)

        return domi_color

    elif H == 56:
        tmp_1 = block[:, 0:14, :, :]
        tmp_2 = block[:, 14:14 + 28, 0:14, :]
        tmp_3 = block[:, 14:14 + 28, -14:, :]
        tmp_4 = block[: -14:, :, :]

    elif H == 84:
        tmp_1 = block[:, 0:14, :, :]
        tmp_2 = block[:, 14:14 + 56, 0:14, :]
        tmp_3 = block[:, 14:14 + 56, -14:, :]
        tmp_4 = block[:, -14:, :, :]


    elif H == 112:
        tmp_1 = block[:, 0:14, :, :]
        tmp_2 = block[:, 14:14 + 84, 0:14, :]
        tmp_3 = block[:, 14:14 + 84, -14:, :]
        tmp_4 = block[:, -14:, :, :]


    tmp_1 = np.reshape(tmp_1, (D, -1, C))
    tmp_2 = np.reshape(tmp_2, (D, -1, C))
    tmp_3 = np.reshape(tmp_3, (D, -1, C))
    tmp_4 = np.reshape(tmp_4, (D, -1, C))


    tmp = np.concatenate((tmp_1, tmp_2, tmp_3, tmp_4), axis=1)

    domi_color = compute_dominant_color_bincount(tmp)

    return domi_color


def compute_pattern_3_color(block_all, idx):

    block = block_all[idx]

    if idx in [0, 3, 5, 6]:
        trans_block_all = []

        for i in range(16):
            cur_block = block[i]

            trans_block = []

            for j in range(3):
                tmp_block = cur_block[:, :, j]
                tmp = tmp_block[np.tril_indices(56)]
                trans_block.append(tmp)

            trans_block = np.array(trans_block)
            trans_block_all.append(trans_block)

    else:

        trans_block_all = []

        for i in range(16):
            cur_block = block[i]

            trans_block = []

            for j in range(3):
                tmp_block = cur_block[:, :, j]
                tmp = tmp_block[np.triu_indices(56)]
                trans_block.append(tmp)

            trans_block = np.array(trans_block)
            trans_block_all.append(trans_block)

    trans_block_all = np.array(trans_block_all) # 16 x 3 x 1596
    trans_block_all = np.transpose(trans_block_all, (0, 2, 1))

    domi_color = compute_dominant_color_bincount(trans_block_all)

    return domi_color


def compute_app_pattern_1(rgb_clip): # 16 x 112 x 112 x 3

    block_all = []

    for m in range(4):
        for n in range(4):
            x_start = m * 28
            x_end = x_start + 28
            y_start = n * 28
            y_end = y_start + 28

            cur_block=rgb_clip[:,x_start:x_end,y_start:y_end,:]
            block_all.append(cur_block)

    ## compute diverse/consitent block idx
    div_idx_1, con_idx_1 = compute_div_cont_block_idx(block_all, pattern=1)

    ## compute diverse/consistent dominant color
    diverse_block = block_all[div_idx_1]
    consist_block = block_all[con_idx_1]

    div_col_1= compute_dominant_color_bincount(diverse_block)
    con_col_1= compute_dominant_color_bincount(consist_block)

    return div_idx_1 + 1, div_col_1 + 1, con_idx_1 + 1, con_col_1 + 1

def compute_app_pattern_2(rgb_clip): # 16 x 112 x 112 x 3

    block_all = []

    block_1 = rgb_clip[:, 42:70, 42:70,:] # 16 x 28 x 28 x 3

    block_2 = rgb_clip[:,28:84,28:84,:]  # 16 x 56 x 56 x 3
    tmp_block = np.zeros_like(block_2)
    tmp_block[:, 14:42, 14:42, :] = rgb_clip[:, 42:70, 42:70, :]
    block_2 = block_2 - tmp_block

    block_3 = rgb_clip[:, 14:98, 14:98, :]  # 16 x 84 x 84 x 3
    tmp_block = np.zeros_like(block_3)
    tmp_block[:, 14:70, 14:70, :] = rgb_clip[:, 28:84, 28:84, :]
    block_3 = block_3 - tmp_block

    block_4 = rgb_clip[:, 0:112, 0:112, :]  # 16 x 84 x 84 x 3
    tmp_block = np.zeros_like(block_4)
    tmp_block[:, 14:98, 14:98, :] = rgb_clip[:, 14:98, 14:98, :]
    block_4 = block_4 - tmp_block

    block_all.append(block_1)
    block_all.append(block_2)
    block_all.append(block_3)
    block_all.append(block_4)

    ## compute diverse/consitent block idx
    div_idx_2, con_idx_2 = compute_div_cont_block_idx(block_all, pattern=2)

    div_col_2 = compute_pattern_2_color(block_all, div_idx_2)
    con_col_2 = compute_pattern_2_color(block_all, con_idx_2)

    return div_idx_2 + 1, div_col_2+1, con_idx_2+1, con_col_2+1


def compute_app_pattern_3(rgb_clip): # 16 x 112 x 112 x 3

    block_all = []

    block_one = rgb_clip[:, 0:56, 0:56, :]
    block_1, block_2 = compute_block_pattern_3(block_one, lu=True)

    block_two = rgb_clip[:, 0:56, 56:112, :]
    block_3, block_4 = compute_block_pattern_3(block_two, lu=False)

    block_three = rgb_clip[:, 56:112, 0:56, :]
    block_5, block_6 = compute_block_pattern_3(block_three, lu=False)

    block_four = rgb_clip[:, 56:112, 56:112, :]
    block_7, block_8 = compute_block_pattern_3(block_four, lu=True)

    block_all.append(block_1)
    block_all.append(block_2)
    block_all.append(block_3)
    block_all.append(block_4)
    block_all.append(block_5)
    block_all.append(block_6)
    block_all.append(block_7)
    block_all.append(block_8)

    block_all = np.array(block_all)

    div_idx_3, con_idx_3 = compute_div_cont_block_idx(block_all, pattern=3)


    div_col_3 = compute_pattern_3_color(block_all, div_idx_3)
    con_col_3 = compute_pattern_3_color(block_all, con_idx_3)


    return div_idx_3+1, div_col_3+1, con_idx_3+1, con_col_3+1



def compute_app_global(rgb_clip):

    global_domi_color = compute_dominant_color_bincount(rgb_clip)


    return global_domi_color+1










