# Water Bottles Classifier

## Project Overview

This repository contains the code, dataset, and documentation for the Water Bottles Classifier project, developed as part of the Data Science course for the 2023-2024 academic year under the guidance of Dr. Ing. Luca Lezzerini (PhD). The aim of this project is to create and evaluate neural network classifiers to identify different brands of water bottles based on their images.

## Project Objectives

1. **Define Classes**: Identify at least three different brands of water bottles.
2. **Collect Data**: Gather a minimum of 100 photos per class, ideally 300 photos per brand, capturing various bottle sizes (e.g., 500 ml, 1 L, 1.5 L).
3. **Build Classifiers**:
   - **Classifier 1**: Neural network with only dense layers, targeting 90% validation accuracy.
   - **Classifier 2**: Neural network with dense and convolutional layers, targeting 93% validation accuracy.
   - **Classifier 3**: Use a pre-trained neural network, targeting 95% validation accuracy.

## Team Composition

The project can be completed individually or in teams of up to four members. The deadline for submission is Monday, June 17, 2024. Late submissions will not be accepted.

## Deliverables

1. **Running Code**: Python scripts or Jupyter notebooks (.py or .ipynb files).
2. **Dataset**: The dataset used for training, in a zipped format.
3. **Report**: A comprehensive report including:
   - Description of the three classifiers
   - Techniques to mitigate overfitting and underfitting, with examples
   - Comparison of the performance of the different classifiers
   - Conclusions drawn from the project

## Sample Images

The dataset includes images of individual products and groups of products. At least three brands are represented in the dataset.

## Repository Structure

```
.
├── data/                       # Dataset used for training
│   └── water_bottles.zip       # Unzip the files before running and put them in the same folder as the zip file
├── notebooks/                  # Jupyter notebooks for the project
│   ├── classifier_dense.ipynb
│   ├── classifier_cnn.ipynb
│   └── classifier_pretrained.ipynb
├── scripts/                    # Python scripts for the project
│   ├── classifier_dense.py
│   ├── classifier_cnn.py
│   └── classifier_pretrained.py
├── report/                     # Project report
│   └── Water_Bottles_Classifier_Report.pdf
└── README.md                   # Project description and instructions
```

## How to Run the Project

1. Clone this repository:
   ```bash
   git clone https://github.com/ergishasani/DataScience_Project
   ```
2. Navigate to the project directory:
   ```bash
   cd water-bottles-classifier
   ```
3. Unzip the dataset:
   ```bash
   unzip data/water_bottles.zip -d data/
   ```
4. Run the desired classifier script or notebook.
#
