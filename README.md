# Pretrained-Simple-YOLOv4-TF

[![license](https://img.shields.io/github/license/mashape/apistatus.svg)](LICENSE)

## Prerequisites
- Tensorflow 2.3.0rc0

## Install requirements

```bash
pip install -r requirements-gpu.txt

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
- You can use command line flag '--image' to specify the input image, and '--output' to specify the output.
```bash
# Object detection in images
python detectimage.py --image InputPath --output OutputPath

# Object detection in videos
python detectvideo.py --video InputPath --output OutputPath

```
- For example:

```bash
# Object detection in images
python detectimage.py --image ./data/kite.jpg --output result.png

# Object detection in videos
python detectvideo.py --video ./data/road.mp4 --output result.avi

```

## References

  * YOLOv4: Optimal Speed and Accuracy of Object Detection [YOLOv4](https://arxiv.org/abs/2004.10934).
  
   My project is inspired by these previous fantastic YOLOv4 implementations:
  * [hunglc007 / tensorflow-yolov4-tflite](https://github.com/hunglc007/tensorflow-yolov4-tflite)
  * [theAIGuysCode / tensorflow-yolov4-tflite](https://github.com/theAIGuysCode/tensorflow-yolov4-tflite)
