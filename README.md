# visdial-tensorflow
This code provides a TensorFlow implementation for [Visual Dialog](https://arxiv.org/pdf/1611.08669.pdf).

## Setup
TensorFlow >= 1.4. Installation instruction are as follows:
```
pip install --user tensorflow-gpu
```
Download and unzip VisDial dataset:
```
https://computing.ece.vt.edu/~abhshkdz/data/visdial/visdial_0.9_train.zip
https://computing.ece.vt.edu/~abhshkdz/data/visdial/visdial_0.9_val.zip
unzip visdial_0.9_train.zip
unzip visdial_0.9_val.zip
```
Dowload COCO dataset from http://cocodataset.org/#download.

## Preprocessing
Use the codes under data/ to preprocess data:

1. prepro.py: preprocess captions, questions, answers and dialog information.
2. resnet152_img.py: extract ResNet-152 feature. [ResNet-152 checkpoint](http://download.tensorflow.org/models/resnet_v1_152_2016_08_28.tar.gz) should be doenloaded at first.	
3. vgg16_img.py: extract VGG-16 feature. [VGG-16 checkpoint](http://download.tensorflow.org/models/vgg_16_2016_08_28.tar.gz) should be doenloaded at first.

