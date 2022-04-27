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
* Train an embedding network of ResNet-50 (D=512) using HIST loss:
```bash
python train.py --gpu-id 0
                --dataset cub
                --model resnet50
                --embedding-size 512
                --tau 32
                --alpha 1.1
                --epochs 40
                --lr 1.2e-4
                --lr-ds 1e-1
                --lr-hgnn-factor 5
                --weight-decay 5e-5
                --lr-decay-step 5
```

### Cars-196
* Train an embedding network of ResNet-50 (D=512) using HIST loss:
```bash
python train.py --gpu-id 0
                --dataset cars
                --model resnet50
                --embedding-size 512
                --tau 32
                --alpha 0.9
                --epochs 50
                --lr 1e-4
                --lr-ds 1e-1
                --lr-hgnn-factor 10
                --weight-decay 1e-4
                --lr-decay-step 10
```

### Stanford Online Products
* Train an embedding network of ResNet-50 (D=512) using HIST loss:
```bash
python train.py --gpu-id 0
                --dataset SOP
                --model resnet50
                --embedding-size 512
                --tau 16
                --alpha 2
                --epochs 60
                --lr 1e-4
                --lr-ds 1e-2
                --lr-hgnn-factor 10
                --weight-decay 1e-4
                --lr-decay-step 10
                --bn-freeze 0
```

## Evaluation
For an evaluation demo, we provide our pre-trained ResNet-50 (D=512) using HIST loss.

* Download our torch models as follows:
```bash
# CUB-200-2011
wget https://github.com/ljin0429/HIST/releases/download/torchmodel/cub_resnet50_best.pth
# Cars-196
wget https://github.com/ljin0429/HIST/releases/download/torchmodel/cars_resnet50_best.pth
# Standord Online Products
wget https://github.com/ljin0429/HIST/releases/download/torchmodel/SOP_resnet50_best.pth
```

* Evaluate the provided pre-trained model or your own trained model:
```bash
# The parameters should be changed according to the model to be evaluated.
python evaluate.py --gpu-id 0 
                   --dataset (cub/cars/SOP) 
                   --model resnet50 
                   --model-path (your_model_path)
```

## Citation


