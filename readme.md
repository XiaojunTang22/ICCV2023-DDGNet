# DDG-Net: Discriminability-Driven Graph Network for Weakly-supervised Temporal Action Localization
[Paper]()

Xiaojun Tang

(**ICCV**),2023

## Table of Contents
1. [Introduction](#introduction)
1. [Preparation](#preparation)
1. [Testing](#testing)
1. [Training](#training)
1. [Citation](#citation)

## Introduction
Weakly-supervised

![avatar](./figs/arch.png)

## Preparation
### Requirements and Dependencies:
Here we list our used requirements and dependencies.
 - Linux: Ubuntu 20.04.4 LTS
 - GPU: NVIDIA RTX A5000
 - CUDA: 11.7
 - Python: 3.8.10
 - PyTorch: 1.11.0
 - Numpy: 1.21.2
 - Pandas: 1.3.5
 - Scipy: 1.7.3 
 - Wandb: 0.12.11
 - Tqdm: 4.61.2

### THUMOS14 Dataset：
We use the 2048-d features provided by MM 2021 paper: Cross-modal Consensus Network for Weakly Supervised Temporal Action Localization. You can get access of the dataset from [here](https://rpi.app.box.com/s/hf6djlgs7vnl7a2oamjt0vkrig42pwho). The annotations are included within this package.

### ActivityNet-v1.2 Dataset：
We also use the features provided in [MM2021-CO2-Net](https://github.com/harlanhong/MM2021-CO2-Net). The features can be obtained from [here](https://rpi.app.box.com/s/hf6djlgs7vnl7a2oamjt0vkrig42pwho). The annotations are included within this package.

## Testing
Download the pretrained models from [Google Drive](https://drive.google.com/drive/folders/1uSUJEo7iN7A3HpY0YD_e_99ECSxk7Tvi?usp=share_link), and put them into "./download_ckpt/".

### Test on THUMOS-14
Change "path/to/CO2-THUMOS-14" in the script into your own path to the dataset, and run:
```
cd scripts/
./test_thumos.sh
```

### Test on ActivityNet-v1.2
Change "path/to/CO2-ActivityNet-12" in the script into your own path to the dataset, and run:
```
cd scripts/
./test_activitynet.sh
```

## Training
Change "path/to/thumos" into your own path to the dataset, and run:
```
./train_thumos.sh
```
Change "path/to/activity" into your own path to the dataset, and run:
```
./train_activity.sh
```


## Citation
If you find the code useful in your research, please cite:

    @inproceedings{mengyuan2022ECCV_DELU,
      author = {Chen, Mengyuan and Gao, Junyu and Yang, Shicai and Xu, Changsheng},
      title = {Dual-Evidential Learning for Weakly-supervised Temporal Action Localization},
      booktitle = {European Conference on Computer Vision (ECCV)},
      year = {2022}
    }

## License

See [MIT License](/LICENSE)

## Acknowledgement

This repo contains modified codes from:
 - [ECCV2022-DELU](https://github.com/MengyuanChen21/ECCV2022-DELU): for implementation of the baseline [DELU (ECCV2022)](https://www.ecva.net/papers/eccv_2022/papers_ECCV/papers/136640190.pdf).

This repo uses the features and annotations from:
 - [MM2021-CO2-Net](https://github.com/harlanhong/MM2021-CO2-Net): [CO2-Net (MM2021)](https://arxiv.org/abs/2107.12589).
