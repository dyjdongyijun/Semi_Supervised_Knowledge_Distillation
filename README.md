# Semi-supervised Relational Knowledge Distillation

## Setup

### Setup teacher models pre-trained on CIFAR-10
We follow the instruction from [PyTorch_CIFAR10](https://vscode.dev/github/huyvnphan/PyTorch_CIFAR10) to download the weights of teacher models pre-trained on CIFAR-10 (or alternatively via [Google Drive](https://drive.google.com/file/d/17fmN8eQdLpq2jIMQ_X0IXDPXfI9oVWgq/view) directly)
```
$ cd ..
$ git clone https://github.com/huyvnphan/PyTorch_CIFAR10.git
$ mv PyTorch_CIFAR10 cifar10_pretrained
$ cd cifar10_pretrained
$ python train.py --download_weights 1
```

----------------

## FixMatch
Based on semi-supervised learning via FixMatch adopted from the following [Unofficial PyTorch implementation of FixMatch](https://github/kekmodel/FixMatch-pytorch)

### Results

#### CIFAR10
| #Labels | 40 | 250 | 4000 |
|:---:|:---:|:---:|:---:|
| Paper (RA) | 86.19 ± 3.37 | 94.93 ± 0.65 | 95.74 ± 0.05 |
| This code | 93.60 | 95.31 | 95.77 |
| Acc. curve | [link](https://tensorboard.dev/experiment/YcLQA52kQ1KZIgND8bGijw/) | [link](https://tensorboard.dev/experiment/GN36hbbRTDaBPy7z8alE1A/) | [link](https://tensorboard.dev/experiment/5flaQd1WQyS727hZ70ebbA/) |

\* November 2020. Retested after fixing EMA issues.
#### CIFAR100
| #Labels | 400 | 2500 | 10000 |
|:---:|:---:|:---:|:---:|
| Paper (RA) | 51.15 ± 1.75 | 71.71 ± 0.11 | 77.40 ± 0.12 |
| This code | 57.50 | 72.93 | 78.12 |
| Acc. curve | [link](https://tensorboard.dev/experiment/y4Mmz3hRTQm6rHDlyeso4Q/) | [link](https://tensorboard.dev/experiment/mY3UExn5RpOanO1Hx1vOxg/) | [link](https://tensorboard.dev/experiment/EDb13xzJTWu5leEyVf2qfQ/) |

\* Training using the following options `--amp --opt_level O2 --wdecay 0.001`

----------------

## Usage

### Train

```
python 
```

----------------

## References
- [Unofficial PyTorch implementation of FixMatch](https://github/kekmodel/FixMatch-pytorch)
- [PyTorch_CIFAR10](https://vscode.dev/github/huyvnphan/PyTorch_CIFAR10)
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
```
