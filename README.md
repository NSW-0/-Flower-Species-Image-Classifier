
# 🌸 Flower Species Image Classifier

A deep learning image classifier built with TensorFlow that identifies 102 different species of flowers using transfer learning with MobileNetV2.

## Overview
This project was completed as part of the **Udacity AI Programming with Python Nanodegree**. The classifier is trained on the [Oxford Flowers 102 dataset](https://www.tensorflow.org/datasets/catalog/oxford_flowers102) and achieves over **76% test accuracy**.

## Features
- Transfer learning using **MobileNetV2** pretrained on ImageNet
- Trained on 102 flower species
- Command line application for easy inference
- Top K predictions with probability scores
- Label mapping from class numbers to flower names

## Project Structure
```
├── notebook.ipynb        # Main training notebook
├── predict.py            # Command line application
├── my_model.keras        # Saved trained model
└── label_map.json        # Class labels to flower names mapping
```

## Usage
**Basic prediction:**
```bash
python predict.py ./test_images/wild_pansy.jpg my_model.keras
```
**Top K predictions:**
```bash
python predict.py ./test_images/wild_pansy.jpg my_model.keras --top_k 5
```
**With flower names:**
```bash
python predict.py ./test_images/wild_pansy.jpg my_model.keras --top_k 5 --category_names label_map.json
```

## Results
| Dataset | Accuracy |
|---------|----------|
| Training | 96.45% |
| Validation | 77.75% |
| Test | 76.35% |

## Technologies
- TensorFlow 2.x
- Keras
- TensorFlow Datasets
- MobileNetV2
- NumPy
- Matplotlib
- PIL
