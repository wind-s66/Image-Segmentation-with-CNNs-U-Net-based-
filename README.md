
# Femur Segmentation with Improved Attention Residual U-Net

## 1. Project Overview

This project focuses on femur segmentation from knee X-ray images using deep learning.

The main goal is to improve the original U-Net architecture and achieve accurate segmentation of the femur region.

The improved model is based on the U-Net design used in DeepPipeline, with additional residual connections and attention mechanisms.

## 2. Model Improvements

The original U-Net architecture was modified to create an Attention Residual U-Net.

The main improvements include:

### 2.1 Residual Blocks

Standard convolutional blocks were replaced with Residual Blocks.

Residual connections allow information to flow directly between layers, helping gradient propagation and feature learning.

### 2.2 Attention Gates

Attention Gates were added to the skip connections between the encoder and decoder.

These gates help the model focus on relevant features while reducing the influence of unnecessary background information.

### 2.3 Spatial Attention

A Spatial Attention module was added to the bottleneck.

It uses average pooling and max pooling across channels to generate spatial attention weights.

This allows the model to emphasize important spatial features.

### 2.4 Dropout

Dropout2d with a rate of 0.2 was added to the bottleneck.

This helps reduce overfitting and improve generalization.

### 2.5 Model Architecture

The improved network consists of:

- Residual Encoder
- Residual Center Block
- Spatial Attention
- Dropout
- Attention-Gated Skip Connections
- Bilinear Upsampling
- Residual Decoder
- Final 1x1 Convolution

The model accepts single-channel grayscale X-ray images and produces single-channel segmentation logits.

## 3. Dataset and Preprocessing

The dataset contains knee X-ray images and manually annotated femur masks.

The dataset is divided into training, validation, and testing sets.

Preprocessing includes:

- Converting X-ray images to grayscale
- Resizing images to 256x256
- Normalizing pixel values to [0, 1]
- Converting segmentation masks to binary values

Training data augmentation includes:

- Random horizontal flipping
- Random rotation between -10 and +10 degrees

The same geometric transformations are applied to images and masks to maintain alignment.

## 4. Training

The model is implemented using PyTorch.

Training configuration:

- Optimizer: Adam
- Learning rate: 0.001
- Batch size: 8
- Epochs: 30
- Base channels: 16
- Loss function: BCEWithLogitsLoss + Dice Loss

The training process calculates:

- Training Loss
- Validation Loss
- Training Dice Score
- Validation Dice Score

The model with the highest validation Dice Score is saved as best.pt.

Training history and learning curves are also saved.

## 5. Evaluation Results

The improved Attention Residual U-Net was evaluated on 19 labeled X-ray images.

Results:

- Mean Dice Score: 0.9760 (97.33%)
- Highest Dice Score:0.9939
- Lowest Dice Score:0.9027
- Number of evaluated images: 20

Most evaluated images achieved Dice Scores above 0.98.

These results demonstrate strong segmentation performance on the evaluated images.

A controlled comparison with the original U-Net using identical training and evaluation conditions is required to confirm the performance improvement.

## 6. Project Structure

    code/
    |-- arg.py
    |-- dataset.py
    |-- model.py
    |-- trainer.py
    |-- evaluate.py
    |-- utils.py
    |-- main.py
    |-- requirements.txt
    |-- CSVs/
    |   |-- train.csv
    |   |-- val.csv
    |   |-- test.csv
    |-- checkpoints/
    |-- results/

File descriptions:

- arg.py: Command-line arguments and configuration
- dataset.py: Dataset loading and augmentation
- model.py: Improved Attention Residual U-Net
- trainer.py: Model training and validation
- evaluate.py: Model evaluation and prediction
- utils.py: Loss functions, metrics, and utilities
- main.py: Main program entry point

## 7. Installation

Install the required Python packages:

    pip install -r requirements.txt

## 8. Usage

Run the following commands from the project directory.

### Train the Model

    python main.py --mode train --epochs 30 --bs 8 --lr 0.001

### Evaluate the Model

    python main.py --mode evaluate --checkpoint checkpoints/best.pt

### Predict Segmentation Masks

    python main.py --mode predict --checkpoint checkpoints/best.pt

Note: The evaluation commands require the updated evaluate.py implementation that accepts command-line arguments through evaluate(args).

The checkpoint must have been trained using the same model architecture.

## 9. Model Outputs

The training process saves:

- best.pt: Best model checkpoint
- last.pt: Final model checkpoint
- history.csv: Training history
- training_curves.png: Training and validation curves

The evaluation process saves:

- Predicted segmentation masks
- Individual Dice Scores
- Mean Dice Score

## 10. Conclusion

This project presents an improved U-Net architecture for femur segmentation.

The main architectural improvements are Residual Blocks, Attention Gates, Spatial Attention, and Dropout.

The improved model achieved a Mean Dice Score of 0.9733 on 19 evaluated X-ray images.

The results indicate that the model can accurately segment the femur region in the evaluated dataset.

## 11. References

DeepPipeline:
https://github.com/lext/deep-pipeline

The improved architecture was developed by adapting the DeepPipeline U-Net design and integrating established residual and attention mechanisms for femur segmentation.
