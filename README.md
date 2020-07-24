# Object-detection

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://github.com/sivaabhishek/Object-detection/blob/master/LICENSE)

You can find the Weights.weights in link  https://pjreddie.com/darknet/yolo/
After the download rename yolov3.weights (236MB) to Weights.weights and place it in the dataset folder.

After that just run the object detection.py and you will be asked to select an image and you will see the output.


In the object detection process, the prediction is done at three scales, which are precisely given by downsampling the dimensions of the input image by 32, 16 and 8 respectively.

The first detection is made by the 82nd layer. For the first 81 layers, the image is down sampled by the network, such that the 81st layer has a stride of 32. If we have an image of 416 x 416, the resultant feature map would be of size 13 x 13.
One detection is made here using the 1 x 1 detection kernel, giving us a detection feature map of 13 x 13 x 255.

![](https://i.ibb.co/GTfnWmB/dog.jpg)

Then, the second detection is made by the 94th layer, yielding a detection feature map of 26 x 26 x 255.

Like before, a few 1 x 1 convolutional layers follow to fuse the information from the previous layer. We make the final of the 3 at 106th layer, yielding feature map of size 52 x 52 x 255.

## Detection of objects of various sizes

The 13 x 13 layer is responsible for detecting large objects, whereas the 52 x 52 layer detects the smaller objects, with the 26 x 26 layer detecting medium objects. Here is a comparative analysis of different objects picked in the same object by different layers.

## Example 1

![](https://serving.photos.photobox.com/00698671d5e6ea9017497487ab132a3b29718d777da910a23b9d10a658efd037bbbf3cbd.jpg)


![](https://serving.photos.photobox.com/14953861b304a2c8ab55f0409c1f30237c7fd520896245424a5c18bd2f5c62b527436cda.jpg)

## Example 2

![](https://i.ibb.co/b7b4NHG/bc.jpg)

![](https://i.ibb.co/YBz286h/opc.jpg)
