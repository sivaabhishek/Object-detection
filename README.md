# Object-detection

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://github.com/sivaabhishek/Object-detection/blob/master/LICENSE)


We apply a single neural network to the full image. This network divides the image into regions and predicts bounding boxes and probabilities for each region. These bounding boxes are weighted by the predicted probabilities. It looks at the whole image at test time so its predictions are informed by global context in the image. It also makes predictions with a single network evaluation unlike systems like R-CNN which require thousands for a single image. This makes it extremely fast, more than 1000x faster than R-CNN.


In the object detection process, the prediction of the class is done at three scales, which are precisely given by downsampling the dimensions of the input image by 32, 16 and 8 respectively.

The first detection is made by the 82nd layer. For the first 81 layers, the image is down sampled by the network, such that the 81st layer has a stride of 32. If we have an image of 416 x 416, the resultant feature map would be of size 13 x 13.
One detection is made here using the 1 x 1 detection kernel, giving us a detection feature map of 13 x 13 x 255.

![](https://i.ibb.co/GTfnWmB/dog.jpg)

Then, the second detection is made by the 94th layer, yielding a detection feature map of 26 x 26 x 255.

Like before, a few 1 x 1 convolutional layers follow to fuse the information from the previous layer. We make the final of the 3 at 106th layer, yielding feature map of size 52 x 52 x 255.

## Detection of objects of various sizes

The 13 x 13 layer is responsible for detecting large objects, whereas the 52 x 52 layer detects the smaller objects, with the 26 x 26 layer detecting medium objects. Here is a comparative analysis of different objects picked in the same object by different layers.

# Instruction to run program

You can find the Weights.weights in link  https://pjreddie.com/darknet/yolo/
After the download rename yolov3.weights (236MB) to Weights.weights and place it in the dataset folder.

After that just run the object detection.py and you will be asked to select an image and you will see the output.

## Example 1

![](https://i.ibb.co/b7b4NHG/bc.jpg)

![](https://i.ibb.co/YBz286h/opc.jpg)
