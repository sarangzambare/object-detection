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


### Training

The training data consists of images labelled with the objects it contains, along with the bounding box for each object. The way this is done is in the form of a vector encoding 6 entities, as given below :

![alt text](https://raw.githubusercontent.com/sarangzambare/object-detection/master/png/trainingdata.png)


where Bx, By represents the centre of the box, and Bh and Bw are the height and width of the box respectively. The class label c is actually a vector of length 80, representing all the 80 classes.

## The YOLO model :

YOLO calculates the probabilities and bounding boxes for objects in a single forward pass.

The way this is done is splitting the image into a 19x19 grid (different grid sizes are possible too) and detecting objects for each of the 19x19 boxes. However, this is not done sequentially, and is actually implemented as a **convolution operation**. Meaning that the individual filter shapes and the depth is chosen in such way that the output is 19x19xchannels.

Having done this, the value in the **i,j** pixel of the output, represents the information about detecting objects in the **i,j** block of the grid.

For example:

If
* Input image size is **(448,448,3)**
* Number of classes to predict = **25**
* Grid size = **(7,7)**

Then, each label vector will be of length 30 (Pc, Bx, By, Bw, Bh, +25 classes).

To have such an output shape, an example architecture will look like :

![alt text](https://raw.githubusercontent.com/sarangzambare/object-detection/master/png/yolo_layers.png)
