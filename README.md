# Hypergraph-Induced Semantic Tuplet Loss for Deep Metric Learning (CVPR 2022)
Official PyTorch implementation of **HIST loss** for deep metric learning | paper (The paper link will be updated soon!)

The repository provides <sup>1)</sup> source codes for the main results and <sup>2)</sup> pre-trained models for quick evaluation.

## Requirements
* Python3
* PyTorch
* [PyTorch Metric Learning](https://github.com/KevinMusgrave/pytorch-metric-learning)
* Numpy
* tqdm
* pandas
* matplotlib
* wandb (optional)

## Installation
We recommend using Conda (or Virtualenv) to set up an environment.

Our implementation was tested on the following libraries with Python 3.6.

* Install PyTorch:
```
pip install torch==1.7.1+cu101 torchvision==0.8.2+cu101 torchaudio==0.7.2 -f https://download.pytorch.org/whl/torch_stable.html
```

* Install the other dependencies:
```
pip install tqdm
pip install pandas
pip install matplotlib
pip install pytorch-metric-learning
pip install wandb
```

## Dataset preparation
Download three public benchmarks for deep metric learning, and extract the tgz or zip files into `./data/`.
* [CUB-200-2011](http://www.vision.caltech.edu/datasets/cub_200_2011/)
* [Cars-196](http://ai.stanford.edu/~jkrause/cars/car_dataset.html) 
* [Stanford Online Products](https://cvgl.stanford.edu/projects/lifted_struct/)

**(Note)** For Cars-196, download both a tar of all images and annotations for both training and test images from the website, and then, put the files into `./data/cars196`.

## Training
Our HIST loss utilizes multilateral semantic relations between every sample and class for a given mini-batch via hypergraph modeling
(see `./code/hist.py`).
By leveraging multilateral semantic relations, HIST loss enables the embedding network to capture important visual semantics suitable for deep metric learning.
A standard embedding network (*e.g.*, ResNet-50) trained with our HIST loss (see `./code/train.py`) achieves SOTA performance on three public benckmarks for deep metric learning.

### CUB-200-2011


### Cars-196


### Stanford Online Products


## Evaluation

## Citation


