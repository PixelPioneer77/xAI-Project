# Explainability of Object Detection Models in Content Moderation

This repository contains a university project focused on the explainability of object detection models, particularly in content moderation. We used the YOLOv8 model and a custom weapon dataset for training.

## Table of Contents
- [Introduction](#introduction)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Dataset](#dataset)
- [Model Training](#model-training)
- [Explainability Methods](#explainability-methods)
- [Results](#results)
- [Contributing](#contributing)
- [License](#license)
- [Acknowledgments](#acknowledgments)

## Introduction

This project aims to improve the transparency and interpretability of object detection models in content moderation by using YOLOv8, trained on a weapon dataset.

## Project Structure

```
├── data
│   ├── raw
│   ├── processed
├── models
│   ├── yolov8
│   ├── trained_models
├── notebooks
│   ├── data_preparation.ipynb
│   ├── model_training.ipynb
│   ├── explainability_analysis.ipynb
├── src
│   ├── data_preparation.py
│   ├── train_model.py
│   ├── explainability.py
├── results
│   ├── charts
│   ├── reports
├── README.md
├── requirements.txt
└── LICENSE
```

## Installation

Clone the repository and install dependencies:

```sh
git clone https://github.com/yourusername/explainability-object-detection.git
cd explainability-object-detection
pip install -r requirements.txt
```

## Dataset

The dataset includes images with weapons and corresponding annotations:

```
data/raw
├── images
├── annotations
```

## Model Training

Training involves data preprocessing, model configuration, and evaluation. Details are in `model_training.ipynb`.

## Explainability Methods

We used:
- Eigen-GradCAM

## Results

Results include:
- Trained model weights
- Performance metrics
- Explainability visualizations

Details are available in the `results` directory and notebooks.

## License

Licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

## Acknowledgments

Thanks to our university, professors, YOLOv8 contributors, and the open-source community.

Feel free to open issues or contact us for questions or feedback. Happy coding!
