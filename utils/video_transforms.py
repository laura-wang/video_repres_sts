import random
import numpy as np


#TODO: operate with different sample lenth

class RandomHorizontalFlip(object):

    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, sample):
        """
        Args:
            img (seq Images): seq Images to be flipped.
        Returns:
            seq Images: Randomly flipped seq images.
        """
        video_clip, u_flow, v_flow = sample['rgb_clip'], sample['u_flow'], sample['v_flow']

        if random.random() < self.p:
            # t x h x w
            #print("flip")
            flip_video_clip = np.flip(video_clip, axis=2).copy()
            flip_u_flow_clip = np.flip(u_flow, axis=2).copy()
            flip_v_flow_clip = np.flip(v_flow, axis=2).copy()


            return {'rgb_clip':flip_video_clip, 'u_flow': flip_u_flow_clip, 'v_flow': flip_v_flow_clip}

        #print("No flip!")

        return sample


class RandomCrop(object):
    """
    Random corp video clip

    """

    def __init__(self, output_size):
        assert isinstance(output_size, (int, tuple))
        if isinstance(output_size, int):
            self.output_size = (output_size, output_size)
        else:
            assert len(output_size) == 2
            self.output_size = output_size


    def __call__(self, sample):

        # video_clip: 16 x 128 x 171 x 3
        # u_flow: 15 x 128 x 171
        # v_flow: 15 x 128 x 171
        video_clip, u_flow, v_flow = sample['rgb_clip'], sample['u_flow'], sample['v_flow']


        h, w = video_clip.shape[1:3]
        new_h, new_w = self.output_size

        h_start = random.randint(0, h-new_h)
        w_start = random.randint(0, w-new_w)

        video_clip = video_clip[:, h_start:h_start+new_h,
                                 w_start:w_start+new_w, :]


        u_flow = u_flow[:, h_start:h_start + new_h, w_start:w_start + new_w]
        v_flow = v_flow[:, h_start:h_start + new_h, w_start:w_start + new_w]



        sample = {'rgb_clip': video_clip, 'u_flow': u_flow, 'v_flow': v_flow}

        return sample



class CenterCrop(object):
    """
    center crop video clip

    """

    def __init__(self, output_size):
        assert isinstance(output_size, (int, tuple))
        if isinstance(output_size, int):
            self.output_size = (output_size, output_size)
        else:
            assert len(output_size) == 2
            self.output_size = output_size

    def __call__(self, sample):
        video_clip, label = sample['rgb_clip'], sample['label']  # video_clip: 16 x 128 x 171 x 2
        h, w = video_clip.shape[1:3]


        new_h, new_w = self.output_size

        h_start = int((h - new_h) / 2)
        w_start = int((w- new_w) / 2)

        video_clip = video_clip[:, h_start:h_start + new_h,
                     w_start:w_start + new_w, :]

        sample = {'rgb_clip': video_clip, 'label': label}

        return sample


class ToTensor(object):

    """
    change input channel
    D x H x W x C ---> C x D x H x w

    """

    def __init__(self):
        super(ToTensor, self).__init__()

    def __call__(self, sample):
        video_clip, u_flow, v_flow = sample['rgb_clip'], sample['u_flow'], sample['v_flow']

        video_clip = np.transpose(video_clip, (3, 0, 1, 2))

        sample = {'rgb_clip': video_clip, 'u_flow': u_flow, 'v_flow': v_flow}

        return sample

