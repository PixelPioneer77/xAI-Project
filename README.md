# Explainability of Object Detection Models in Content Moderation

This repository contains a university project focused on the explainability of object detection models, particularly in content moderation. We used the YOLOv8 model and a custom weapon dataset for training.

## Introduction

This project aims to improve the transparency and interpretability of object detection models in content moderation by using YOLOv8, trained on a weapon dataset.

## Dataset
For training our custom YOLOv8 model we used the dataset from https://github.com/ari-dasci/OD-WeaponDetection which includes weapons and similar objects.

## Model Training

Training involves data preprocessing, model training, and evaluation. Details are in the `YOLO training` folder.
In addition to that we also used an already pre-trained weapons model from https://github.com/JoaoAssalim/Weapons-and-Knives-Detector-with-YOLOv8/tree/main

## Explainability Methods
We used the following repo for the explainability part https://github.com/Spritan/YOLOv8_Explainer

We used:
- Eigen-GradCAM

## Results

Results include:
- Trained model weights
- Performance metrics
- Explainability visualizations
- POC with GUI

Details are available in the `explainability_poc.ipynb` notebook.


## Acknowledgments

Thanks to our university, professors, YOLOv8 contributors, and the open-source community.

## GitHub Repo
https://github.com/PixelPioneer77/xAI-Project