# Image Segmentation using RCNN

This repository provides basic insight into building an R-CNN (Regions with Convolutional Neural Network features) model from scratch. The concept of an R-CNN has been built on extensively to create modern computer vision techniques, and updated versions of R-CNN models (e.g., [Faster R-CNN](https://arxiv.org/abs/1506.01497)) can be used with popular machine learning libraries like [Pytorch](https://pytorch.org/vision/stable/models.html#id35).

An R-CNN is simply a CNN that performs object classification on regions of images created from a single image. For this example, OpenCV's selective search segmentation algorithm is used to pick out region proposals. In general, a selective search segmentation algorithm operates by recursively joining similar neighboring. The joined pixels are called regions, and the algorithm finishes when each remaining region is too dissimilar from any of its neighbors to be joined. The similarities taken into consideration are attributes like color, texture, size, and shape. An example of the regions created by running selective search can be seen below.

<p align="center">
    <img src="/images/no_nms.png" | width=650>
</p>

Once the region proposals have been generated, a pre-trained CNN (this example uses ResNet18) predicts the label for each sub-image denoted by the region. Since hundreds (or even thousands) of regions are created in this process, certain regions are thrown out if the class's prediction lies below a specified threshold. In this example, a threshold of 90% confidence in the chosen class label is used. In a larger example, it might be feasible to train a background class and throw out any regions that have a high probability of belonging to that class. Further reduction of the regions is done using [non-maximum supression](https://pytorch.org/vision/stable/ops.html#torchvision.ops.nms) (NMS), which essentially eliminates regions based on their intersection-over-union (IoU) score. This technique removes regions which had a lower probability for the class label they predict if their IoU is greater than a certain threshold. What this means is that if two regions overlap and predict the same label, the region with the lower probability will be thrown out because they are likely trying to predict on the same object. After NMS has been applied, you can see that the number of region proposals has dropped drastically:

<p align="center">
    <img src="/images/post_nms.png" | width=650>
</p>