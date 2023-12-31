# Project Title: Deep Learning Model for Predicting RNA Binding Intensity

## Overview

This project aims to build a deep learning model for predicting RNA binding intensity using Convolutional Neural Networks (CNN). The model is trained on the RNAcompete dataset based on RNA Bind-n-Seq (RBNS) data. The goal is to predict the binding intensity of RNA sequences, with a focus on computational biology applications.

## Features

- **Architecture:** The deep learning model employs a Convolutional Neural Network (CNN) architecture.
  
- **Input Representation:** RNA sequences are transformed into one-hot encoded matrices to serve as input to the model.

- **Loss Function:** The loss function used for training is negative Pearson correlation, defined as `pearson_loss`.

- **Dynamic Dropout:** A custom callback, `IncreaseDropoutCallback`, is implemented to dynamically increase the dropout rate for the fully connected layer during training.

- **Preprocessing:** RBNS data is preprocessed to create labels based on concentration and occurrences. Additionally, RNAcompete sequences are transformed into one-hot encoded matrices.

## Hyperparameters

- **Number of Filters:** 32
- **Kernel Size:** 6
- **Fully Connected Layer Size:** 16
- **Initial Dropout Rate:** 0.2
- **Dropout Rate Increase:** 0 (initially)
- **Dropout Increase Epoch:** 3
- **Learning Rate:** 0.05
- **Mini-Batch Size:** 512
- **Number of Epochs:** 1

## Usage

To run the project, execute the Python script with the following command:

```bash
python dnabind.py output_file rna_comp_file input_file rbns_file1 rbns_file2 ... rbns_file
```
- **output_file:** File to save the predicted results.
- **rna_comp_file:** RNAcompete dataset file.
- **input_file:** RBNS input file.
- **rbns_file1, ..., rbns_fileN:** RBNS concentration files.
## Dependencies
- TensorFlow
- Keras
- NumPy
Make sure to install these dependencies using:
```
pip install tensorflow keras numpy
```
## Author
Gilad Aharoni

## Acknowledgments
This project was developed as part of Deep Learning in Computational Biology course By Prof. Yaron Ornstein, Bar Ilan University.
