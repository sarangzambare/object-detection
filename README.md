# Object-detection using custom filtering for the YOLOv2 model, with COCO - classes.


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

### Anchor boxes :

Anchor boxes are predefined boxes of fixed height and width. The idea is to use a finite number of anchor boxes, such that any object detected fits snugly inside at least one of the predefined boxes. The reason this is done is to limit the number of possible values of **Bh** and **Bw** (infinite) to only a few predefined values.

Anchor boxes are chosen based on application. The crux of the idea is that different objects fall in different ratios of height and width. For example, if the classes in this program only had "person" and "car" , we could've used only two anchor boxes :

![alt text](https://raw.githubusercontent.com/sarangzambare/object-detection/master/png/car_man.png)


This approach reduces computational cost, which is of essence particularly in real-time detection.


### Visualizing YOLO :

In this program, I am using an input images which are preprocessed into being 608x608 pixels. The grid size I am using is 19x19. Hence, the following specifications:

* Input shape is **(m,608,608,3)**, where **m** is the number of images per batch.
* Number of anchor boxes = **5**
* Number of classes to predict = **80**
* Output shape is therefore **(m,19,9,5,85)**

If the output shape is confusing, think about it as returning 5 boxes per block, and each of the 5 boxes has a 85 length vector which consists of [Pc,Bx,By,Bh,Bw, + 80 classes]

Graphically :

![alt text](https://raw.githubusercontent.com/sarangzambare/object-detection/master/png/yolo_arch.png)


To predict objects and their bounding boxes,

* For each of the 361 boxes (19<sup>2</sup>), we calculate probability scores by multiplying Pc by the individual class probabilities.

![alt text](https://raw.githubusercontent.com/sarangzambare/object-detection/master/png/equation_c.png)

![alt text](https://raw.githubusercontent.com/sarangzambare/object-detection/master/png/equation_p.png)

![alt text](https://raw.githubusercontent.com/sarangzambare/object-detection/master/png/class_score.png)

After doing this, if we colour each of the 361 blocks with its predicted class, it gives us a nice way to visualize what's being predicted :

![alt text](https://raw.githubusercontent.com/sarangzambare/object-detection/master/png/class_colours.png)

Another way to visualize what's being predicted is to plot the bounding boxes themselves. Without any filtering, there is a large number of boxes, with many boxes pertaining to the same object :

![alt text](https://raw.githubusercontent.com/sarangzambare/object-detection/master/png/class_boxes_2.png)
