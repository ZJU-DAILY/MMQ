## MMQ

This repository contains the code used in our paper: Finding Materialized Models for Model Reuse

We upload all the source code about MMQ in three folders, i.e., Image Classification, Text Classification, and Image Regression. We also provide an experimental case in the folder Exp-Case.

## Environment

Python 3.6.12 

PyTorch 1.6.0

## Dataset

ImageNet-2012 : Download from https://image-net.org/challenges/LSVRC/2012/2012-downloads.php

CIFAR-10 : Download from https://pytorch.org/vision/stable/datasets.html#cifar

FashionMNIST : Download from https://pytorch.org/vision/stable/datasets.html#fashion-mnist

IMDB : Download from http://ai.stanford.edu/~amaas/data/sentiment/index.html

RCV1 : Download from http://riejohnson.com/cnn_data.html

House Price : Download from https://github.com/emanhamed/Houses-dataset

## Run Experimental Case

1.  Run the command in the terminal and get the separation degree of all materialized models

    ```shell
    python MMQ_FashionMNIST2CIFAR10
    ```

2.  Retrain all the models on target dataset and get accuracy of all materialized models on the target dataset

    ```shell
    python retrain_FashionMNIST2CIFAR10
    ```
