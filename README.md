# Pretrained-Simple-YOLOv4-TF

[![license](https://img.shields.io/github/license/mashape/apistatus.svg)](LICENSE)

## Introduction

YOLOv4 Tensorflow implementation for all newbies. Quick access to object detection for applications without AI knowledge. Ready for use with pretrained weights (the original Darknet weights from [AlexeyAB](https://github.com/AlexeyAB/darknet)). Minimal code and easy to follow.

Once you are comfortble with the code, you can either choose to extend the functionality, such as OCR (Optical Character Recognition), Object Tracking, Object Counting, Object Detection for Webcam, etc., check out [theAIGuysCode](https://github.com/theAIGuysCode) for inspirations. Or you can choose to customize the configuration parameters, migrate to TensorFlow Lite for mobile devices, fine-tune weights for your domain, or even transfer learning for other purpose, check out [hunglc007 / tensorflow-yolov4-tflite](https://github.com/hunglc007/tensorflow-yolov4-tflite) for inspirations.

## Code map
```bash

core
  |________config.py                 Configuration for YOLOv4
  |________utils.py                  Utilities for YOLOv4
data
  |________classes
  |          |________coco.names     List of object types for YOLOv4  
  |________kite.jpg                  Example image
  |________road.mp4                  Example video
yolov4-416                           Pretrained weights
detectimage.py                       Object detection for image
detectvideo.py                       Object detection for video
requirements-cpu.txt                 Requirements for running with CPU
requirements-gpu.txt                 Requirements for running with GPU
result.png                           Example output image
results.avi                          Example output video

```
## Demo for running in cloud (Google Colab)
[Pretrained-Simple-YOLOv4-TF.ipynb](https://drive.google.com/file/d/16ygyMRJwKeVDFI5IfEeH_e146gw1Ygad/view?usp=sharing) 

## Prerequisites
- Tensorflow 2.3.0rc0

## Install requirements

```bash
pip install -r requirements-gpu.txt

```
or

```bash
pip install -r requirements-cpu.txt

```

## Unzip pretrained weights

```bash
# Go to the directory containing weights
cd /content/Pretrained-Simple-YOLOv4-TF/yolov4-416/variables/

# Concatenate the zip parts into a whole one
cat variables.z* > variables-all.zip

# Unzip for the weights
unzip variables-all.zip

```

## Detect objects in images or videos
<img src="https://github.com/yuhang2685/Pretrained-Simple-YOLOv4-TF/blob/main/data/kite.jpg" width="35%"><img src="https://github.com/yuhang2685/Pretrained-Simple-YOLOv4-TF/blob/main/result.png" width="35%">

```bash
# Object detection in images
python detectimage.py --image ./data/kite.jpg --output result.png

# Object detection in videos
python detectvideo.py --video ./data/road.mp4 --output result.avi

```
### TODO
* [ ]  Supply the demo for running in cloud (Colab)
* [ ]  Add results for illustration
* [ ]  Supply the code map


## References

  * YOLOv4: Optimal Speed and Accuracy of Object Detection [YOLOv4](https://arxiv.org/abs/2004.10934).
  
   My project is based on these previous fantastic YOLOv4 implementations:
  * [hunglc007 / tensorflow-yolov4-tflite](https://github.com/hunglc007/tensorflow-yolov4-tflite)
  * [theAIGuysCode / tensorflow-yolov4-tflite](https://github.com/theAIGuysCode/tensorflow-yolov4-tflite)
