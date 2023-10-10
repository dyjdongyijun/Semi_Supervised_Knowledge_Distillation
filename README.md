# Cluster-aware Semi-supervised Learning: DAC + RKD

This repository presents code for our [paper](https://arxiv.org/abs/2307.11030):
```
Cluster-aware Semi-supervised Learning: Relational Knowledge Distillation Provably Learns Clustering.
Yijun Dong*, Kevin Miller*, Qi Lei, Rachel Ward. NeurIPS 2023.
```

## Setup

### Environment setup
```
$ conda env create -f environment.yml -n rkd
$ conda activate rkd
```

### File organization
- The relative paths for the datasets, the pretrained teacher models/features, and the pre-allocated directory for results are configured as follows.
    ```python
    ..
    |-- cifar10_pretrained # teacher models pretrained on CIFAR-10 
    |-- data # datasets: cifar-10/100
    |-- pretrained # teacher models
    |   |-- cifar10
    |   |   |-- densenet161_cifar10_dim10.pt # pretrained teacher features
    |   |   |-- densenet161_cifar1010_active-fl_40.npy # coreset labeled samples selected via StochasticGreedy
    |   |   |-- ...
    |   |-- cifar100
    |   |   |-- resnet50w5_swav_dim1000.pt # pretrained teacher features
    |   |   |-- resnet50w5_swav1000_active-fl_400.npy # coreset labeled samples selected via StochasticGreedy
    |   |   |-- ...
    |-- result # experiment results
    |-- Semi_Supervised_Knowledge_Distillation # main implementation (this repo)
    ```

### Pretrained teacher features
- **CIFAR-10 pretrained models**:
    For CIFAR-10 experiments, we use in-distribution teacher models pretrained on the same dataset (CIFAR-10) based on the [PyTorch models trained on CIFAR-10 dataset](https://github.com/huyvnphan/PyTorch_CIFAR10.git) as follows:
    ```
    $ cd ..
    $ git clone https://github.com/huyvnphan/PyTorch_CIFAR10.git
    $ mv PyTorch_CIFAR10 cifar10_pretrained
    $ cd cifar10_pretrained
    $ python train.py --download_weights 1
    ```
    Alternatively, one can download the pretrained weights directly from the [Google Drive](https://drive.google.com/file/d/17fmN8eQdLpq2jIMQ_X0IXDPXfI9oVWgq/view) link provided in [PyTorch_CIFAR10](https://github/huyvnphan/PyTorch_CIFAR10) and unzip the file in `../cifar10_pretrained/cifar10_models/`.

- **CIFAR-100 pretrained models**: 
    For CIFAR-100 experiments, we use out-of-distribution teacher models pretrained on a different dataset (ImageNet) via (unsupervised) contrastive learning (SwAV) based on the official [PyTorch implementation and pretrained models for SwAV](https://github.com/facebookresearch/swav) as follows:
    ```python
    import torch
    model = torch.hub.load('facebookresearch/swav:main', 'resnet50w5')
    ```

- **Inference of teacher features on the pretrained teacher models**
    ```
    $ bash teach.sh
    ```
    For CIFAR-10 features evaluated with supervisedly pretrained DenseNet161 on CIFAR-10:
    ```
    $ python teach.py --dataset cifar10 --teacher_arch densenet161 --teacher_pretrain cifar10
    ```
    For CIFAR-100 features evaluated with pretrained ResNet-50 (of width x5, i.e., resnet50w5) on ImageNet via SwAV:
    ```
    $ python teach.py --dataset cifar100 --teacher_arch resnet50w5 --teacher_pretrain swav
    ```

### FixMatch
- We follow the implementations in [fbuchert: Unofficial PyTorch implementation of FixMatch](https://github.com/fbuchert/fixmatch-pytorch) and [kekmodel: Unofficial PyTorch implementation of FixMatch](https://github/kekmodel/FixMatch-pytorch)

----------------

## Experiments
- CIFAR-10 experiments
    ```
    $ bash train_cifar10.sh
    ```
- CIFAR-100 experiments
    ```
    $ bash train_cifar100.sh
    ```

----------------

## References
- [fbuchert: Unofficial PyTorch implementation of FixMatch](https://github.com/fbuchert/fixmatch-pytorch)
- [kekmodel: Unofficial PyTorch implementation of FixMatch](https://github/kekmodel/FixMatch-pytorch)
- [PyTorch models trained on CIFAR-10 dataset](https://github.com/huyvnphan/PyTorch_CIFAR10.git) 
- [PyTorch implementation and pretrained models for SwAV](https://github.com/facebookresearch/swav)
- [Official TensorFlow implementation of FixMatch](https://github.com/google-research/fixmatch)
- [Unofficial PyTorch implementation of MixMatch](https://github.com/YU1ut/MixMatch-pytorch)
- [Unofficial PyTorch Reimplementation of RandAugment](https://github.com/ildoonet/pytorch-randaugment)
- [PyTorch image models](https://github.com/rwightman/pytorch-image-models)

## Citations
```
@article{sohn2020fixmatch,
    title={FixMatch: Simplifying Semi-Supervised Learning with Consistency and Confidence},
    author={Kihyuk Sohn and David Berthelot and Chun-Liang Li and Zizhao Zhang and Nicholas Carlini and Ekin D. Cubuk and Alex Kurakin and Han Zhang and Colin Raffel},
    journal={arXiv preprint arXiv:2001.07685},
    year={2020},
}

@article{caron2020unsupervised,
  title={Unsupervised Learning of Visual Features by Contrasting Cluster Assignments},
  author={Caron, Mathilde and Misra, Ishan and Mairal, Julien and Goyal, Priya and Bojanowski, Piotr and Joulin, Armand},
  booktitle={Proceedings of Advances in Neural Information Processing Systems (NeurIPS)},
  year={2020}
}
```
