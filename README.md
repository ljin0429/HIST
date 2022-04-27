# Hypergraph-Induced Semantic Tuplet Loss for Deep Metric Learning (CVPR 2022)
Official PyTorch implementation of **HIST loss** for deep metric learning | paper

## About
Our HIST loss utilizes multilateral semantic relations between every sample and class for a given mini-batch via hypergraph modeling.

By leveraging multilateral semantic relations, HIST loss enables the embedding network to capture important visual semantics suitable for deep metric learning.

A standard embedding network (*e.g.*, ResNet-50) trained with our HIST loss achieves SOTA performance on three public benckmarks for deep metric learning.

The repository provides <sup>1)</sup> source codes for the main results and <sup>2)</sup> pre-trained models for quick evaluation.

## Requirements
* Python3
* PyTorch 1.7.1
* [PyTorch Metric Learning](https://github.com/KevinMusgrave/pytorch-metric-learning)
* Numpy
* tqdm
* pandas
* matplotlib
* wandb (optional)

## Installation
