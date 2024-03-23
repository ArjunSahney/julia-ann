# MNIST Handwritten Digit Recognition using Convolutional Neural Networks

This repository contains a Julia implementation of a Convolutional Neural Network (CNN) for handwritten digit recognition using the MNIST dataset. The code demonstrates how to load the dataset, preprocess the data, define the CNN architecture, train the model, evaluate its performance, and make predictions on new data.

## Dataset

The MNIST dataset is a widely used benchmark dataset for image classification tasks. It consists of 70,000 grayscale images of handwritten digits (0-9), with each image being 28x28 pixels in size. The dataset is split into 60,000 training images and 10,000 test images.

## Requirements

To run the code in this repository, you need to have the following dependencies installed:

- Julia (version 1.0 or higher)
- MLDatasets.jl
- Flux.jl
- ProgressMeter.jl
- BSON.jl
- Images.jl

You can install these packages using the Julia package manager by running the following commands in the Julia REPL:

```julia
using Pkg
Pkg.add("MLDatasets")
Pkg.add("Flux")
Pkg.add("ProgressMeter")
Pkg.add("BSON")
Pkg.add("Images")
```

## Usage

1. Clone this repository to your local machine or download the source code files.

2. Open a terminal and navigate to the directory where the code files are located.

3. Run the Julia script using the following command:
   ```
   julia mnist_digit_recognition.jl
   ```

4. The script will start training the CNN model on the MNIST dataset. The training progress will be displayed in the terminal, showing the epoch number, training accuracy, and test accuracy.

5. After training, the script will save the trained model to a file named `mnist_model.bson` in the same directory.

6. The script will then load the trained model and evaluate its performance on the test set, printing the test loss and accuracy.

7. Finally, the script will make predictions on a subset of test images and display the predicted labels alongside the ground truth labels.

## Model Architecture

The CNN architecture used in this code consists of the following layers:

- Convolutional layer with 16 filters of size 3x3
- Max pooling layer with a pool size of 2x2
- Convolutional layer with 32 filters of size 3x3
- Max pooling layer with a pool size of 2x2
- Convolutional layer with 64 filters of size 3x3
- Max pooling layer with a pool size of 2x2
- Flatten layer
- Fully connected layer with 128 units and ReLU activation
- Fully connected layer with 10 units and softmax activation

## Results

The trained CNN model achieves high accuracy on the MNIST test set, typically around 98-99% accuracy.

## Visualization

The script also includes code to visualize the learned filters of the first convolutional layer. After training, the script saves the filters as grayscale images in the same directory with the names `filter_1.png`, `filter_2.png`, and so on.

Feel free to explore and modify the code to experiment with different architectures, hyperparameters, and visualization techniques!
