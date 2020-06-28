import os
import cv2
import numpy as np

## 16 frames
def load_rgb(rgb_dir, clip_len, start_frame, output_size):

    assert isinstance(output_size, (int, tuple))
    if isinstance(output_size, int):
        resize_height = output_size
        resize_width = output_size
    else:
        assert len(output_size) == 2
        resize_height, resize_width = output_size

    video_clip = []

    for i in range(clip_len):
        cur_img_path = os.path.join(rgb_dir,
                                     'frame'+"{:06}.jpg".format(start_frame + i))
        img = cv2.imread(cur_img_path)
        if img is None:
            print(cur_img_path)
        img = cv2.resize(img, (resize_width, resize_height))


        video_clip.append(img)

    video_clip = np.array(video_clip)

    return video_clip


def load_flow(flow_dir, clip_len, start_frame, output_size):
    assert isinstance(output_size, (int, tuple))
    if isinstance(output_size, int):
        resize_height = output_size
        resize_width = output_size
    else:
        assert len(output_size) == 2
        resize_height, resize_width = output_size

    flow_15 = []

    for i in range(clip_len):
        cur_img_path = os.path.join(flow_dir, 'frame'+'{:06}.jpg'.format(start_frame + i))
        img = cv2.imread(cur_img_path)
        img = img[..., 0]
        img = cv2.resize(img, (resize_width, resize_height))

        flow_img = img.astype(np.float32)
        flow_img = ((flow_img * 40.) / 255.) - 20

        flow_15.append(flow_img)

    flow_15 = np.array(flow_15)

    return flow_15


