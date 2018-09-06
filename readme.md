#Object Detection and Tracking

## Overview
Object Detection Project for Enrich-AI. The module first initializes a detector and a tracker. Next, detector localizes the vehicles in each video frame. The tracker is then updated with the detection results. Finally the tracking results are annotated and displayed in a video frame.

## Detection
In this project, object detection takes a video as input and produces the bounding boxes as the output.TensorFlow Object Detection API is used, which is an open source framework built on top of TensorFlow to construct, train and deploy object detection models. The Object Detection API also comes with a collection of detection models pre-trained on the COCO dataset that are well suited for fast prototyping. In this project, the model: ssd\_mobilenet\_v1\_coco is used that is based on Single Shot Multibox Detection (SSD) framework with minimal modification.

The detector is implemented in ```ObjectDetector``` class in detector.py.

## Kalman Filter for Bounding Box Measurement

Kalman filter is used for tracking objects. Kalman filter has the following important features that tracking can benefit from:

* Prediction of object's future location
* Correction of the prediction based on new measurements
* Reduction of noise introduced by inaccurate detections
* Facilitating the process of association of multiple objects to their tracks

Kalman filter consists of two steps: prediction and update. The first step uses previous states to predict the current state. The second step uses the current measurement, such as detection bounding box location , to correct the state.

## Pipeline

There are two important design parameters, ```min_hits``` and ```max_age```, in the pipe line.  The parameter ```min_hits``` is the number of consecutive matches needed to establish a track. The parameter ```max_age``` is number of consecutive unmatched detection before a track is deleted. Both parameters need to be tuned to improve the tracking and detection performance.

## Issues

The main issue is occlusion. For example, when multiple people are close to each other. This can fool the detector to output a single(and bigger bounding) box, instead of separate bounding boxes. In addition, the tracking algorithm may treat this detection as a new detection and set up a new track.


