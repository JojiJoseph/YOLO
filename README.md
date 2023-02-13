# YOLO

It's my attempt to implement [yolo v1](https://arxiv.org/abs/1506.02640) from scratch.

## How to reproduce the results

 - Create a folder named `dataset`.

 - Download pascal VOC dataset (https://www.kaggle.com/datasets/zaraks/pascal-voc-2007) and copy the folders `Annotations` and `JPEGImages` from one of the trainval folder to the newly created `dataset` folder.

 - Run `python3 preprocess.py`.  It will generate two additional folders with preprocessed data.

 - Run `python3 train.py` for training the model.

 - Run `python3 demo.py` to see realtime demo with a webcam. (You can download the weights from https://github.com/JojiJoseph/YOLO/releases/download/v0.0.0-alpha/model.h5 if you don't want to train it from scratch.)

## Changes from the paper

 - Original backbone network is inspired by GoogLeNet. Here I used InceptionV3 as backbone.
 - Instead 7x7 output grid, I used 5x5 grid
 - Original authors trained their model with 448x448 resolution images with 64 images per batch. I trained 224x224 resolution images with 16 images per bactch.
 - I have used a pseudo iou measurement instead of actual iou for training. Will retrain with actual ios later.
 - In original paper two bounding boxes are predicted. Here I am predicting only one bounding box.



