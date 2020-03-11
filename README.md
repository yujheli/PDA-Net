![Python 3](https://img.shields.io/badge/python-3-green.svg) 
# Cross-Dataset Person Re-Identification via Unsupervised Pose Disentanglement and Adaptation

<p align="center"><img src='model.png' width="1000px"></p>

[[Paper]](http://openaccess.thecvf.com/content_ICCV_2019/papers/Li_Cross-Dataset_Person_Re-Identification_via_Unsupervised_Pose_Disentanglement_and_Adaptation_ICCV_2019_paper.pdf)
Pytorch implementation for our ICCV 2019 paper.

## Prerequisites
- Python 3
- [Pytorch](https://pytorch.org/) (We run the code under version 0.4.1)

## Getting Started

### Installation

- Clone this repo:
```
git clone https://github.com/yujheli/PDA-Net
cd CPD-GAN/
```
- Install dependencies. You can install all the dependencies by:
```
pip install -r requirement.txt
```

### Datasets
We conduct experiments on [Market1501](http://www.liangzheng.org/Project/project_reid.html), [DukeMTMC-reID](https://github.com/layumi/DukeMTMC-reID_evaluation), [CUHK03](http://www.ee.cuhk.edu.hk/~xgwang/CUHK_identification.html) datasets. We need pose landmarks for each dataset during training, so we generate the pose files by [Realtime Multi-Person Pose Estimation](https://github.com/tensorboy/pytorch_Realtime_Multi-Person_Pose_Estimation). And the raw datasets have been preprocessed by the code in [open-reid](https://github.com/Cysu/open-reid). 



## Acknowledgements
Our code is HEAVILY borrowed and modified from [FD-GAN](https://github.com/yxgeee/FD-GAN)and [open-reid](https://github.com/Cysu/open-reid).
