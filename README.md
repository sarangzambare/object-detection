# object-detection using custom filtering for the YOLOv2 model, with COCO - classes.


YOLOv2 (You only look once) is one of the most popular algorithms for object detection. As the name implies, the predictions of objects, and their bounding boxes are calculated as a **single forward pass** through the convolutional neural network, making it suitable for **real time** object detection.


In this repository, I make custom preprocessing methods to be operated on the output of the YOLOv2, to detect common objects encountered while driving in an urban environment.

## What objects are we detecting ?

This program uses the [COCO (Common Objects in Context)](http://cocodataset.org/#home) class list, which has 80 object categories. For reference, these classes are given in the file "coco_classes.txt", and the first few entries are :

```
person
bicycle
car
motorbike
aeroplane
bus
train
truck
boat
traffic light
fire hydrant
stop sign
parking meter
bench
bird
cat
dog
horse
sheep
cow
...
```
