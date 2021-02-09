# Self-Supervised Video Reprepresentation Learning by Uncovering Spatio-temporal Statistics
Pytroch implementation of our T-PAMI 2021 paper "[Self-Supervised Video Reprepresentation Learning by Uncovering Spatio-temporal Statistics](https://arxiv.org/pdf/2008.13426.pdf)", a journal extension of our preliminary work presented in [CVPR 2019](http://openaccess.thecvf.com/content_CVPR_2019/html/Wang_Self-Supervised_Spatio-Temporal_Representation_Learning_for_Videos_by_Predicting_Motion_and_CVPR_2019_paper.html). Extensive additional ananlysis are presented in this journal version. The performance is also significantly improved, by nearly 30%.

The Tensorflow implementation (of our previous CVPR 2019 work) is available at: https://github.com/laura-wang/video_repres_mas.

# Overview
Framework of the proposed approach.

Given an unlabeled video clip, 14 motion statistical labels and 13 appearance statistical labels are to be regeressed. These labels characterize the spatial location and dominant direction of the largest motion, the spatial location and dominant color of the largest color diversity along the temporal axis, etc. 

<p align="center">
  <img src="https://s1.ax1x.com/2020/06/28/N2krnO.md.png" />
</p>

# Requirements
- pytroch >= 1.3.0
- tensorboardX
- cv2
- scipy

# Usage

## Data preparation

UCF101 dataset
- Download the original UCF101 dataset from the [official website](https://www.crcv.ucf.edu/data/UCF101.php). And then extarct RGB images from videos and finally extract optical flow data using TVL1 method.
- Or direclty download the pre-processed RGB and optical flow data of UCF101 [here](https://github.com/feichtenhofer/twostreamfusion) provided by feichtenhofer.

## Train

`python train.py --rgb_prefix RGB_DIR --flow_x_prefix FLOW_X_DIR --flow_y_prefix FLOW_Y_DIR`

## TODO
Feature evaluation

- [ ] Video Retrieval
- [ ] Dynamic Scene Recognition
- [ ] Action Similarity Labeling

## Citation

If you find this repository useful in your research, please consider citing:

```
@Article{wang2021self,
author = {Jiangliu Wang, Jianbo Jiao, Linchao Bao, Shengfeng He, Yunhui Liu, Wei Liu},
title = {Self-Supervised Video Representation Learning by Uncovering Spatio-temporal Statistics},
journal = {IEEE Transactions on Pattern Analysis and Machine Intelligence},
year = {2021}
}
```

```
@inproceedings{wang2019self,
  title={Self-Supervised Spatio-Temporal Representation Learning for Videos by Predicting Motion and Appearance Statistics},
  author={Wang, Jiangliu and Jiao, Jianbo and Bao, Linchao and He, Shengfeng and Liu, Yunhui and Liu, Wei},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  pages={4006--4015},
  year={2019}
}
```






