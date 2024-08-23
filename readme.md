# Machine Learning Project: MLPs and CNNs from Scratch

This project demonstrates the implementation and application of Multilayer Perceptrons (MLPs) and Convolutional Neural Networks (CNNs) for image classification tasks using the PyTorch library. It provides hands-on experience building and training these models on the MNIST dataset.

## Key Features

- **MLPs and CNNs:** Implements MLPs and CNNs from scratch, showcasing their architectures and forward passes.
- **MNIST Dataset:** Utilizes the MNIST dataset for handwritten digit recognition.
- **PyTorch:** Leverages PyTorch for model definition, loss calculation, and gradient-based optimization.
- **K-fold Cross-Validation:** Employs K-fold cross-validation for hyperparameter tuning.
- **Training and Testing:** Includes training and testing loops for evaluating model performance.
- **Visualization:** Visualizes training curves, filtered images, and convolutional filters.

## Requirements

- **PyTorch:** Install PyTorch using the instructions at [https://pytorch.org/get-started/locally/](https://pytorch.org/get-started/locally/).
- **Other Libraries:** The code also imports NumPy, Matplotlib, and tqdm. These can be installed using `pip install numpy matplotlib tqdm`.

## Usage

1. **Install Requirements:** Make sure you have PyTorch and other necessary libraries installed.
2. **Run the Code:** Execute the Python code provided in the project.
3. **Observe:** The code will train and test MLP and CNN models on the MNIST dataset. You'll see:
   - Training curves (loss vs. number of images trained on) for both models.
   - Example images with their predicted labels before and after training.
   - Visualizations of the filtered images and convolutional filters learned by the CNN.
   - Test accuracy and average loss for both models.

## Code Structure

- **Data Loading:** The `load_mnist` function loads the MNIST dataset using PyTorch's `torchvision.datasets.MNIST` and creates data loaders for training and testing.
- **Models:** 
   - `MyPerceptron`: Defines a simple perceptron using a single linear layer.
   - `MyMLP`: Defines a multilayer perceptron with multiple linear layers and ReLU activation functions.
   - `MyCNN`: Defines a convolutional neural network with convolutional layers, ReLU activations, and a final MLP for classification.
- **Training and Testing:** 
   - `training`: Trains a given model on the training data using a specified loss function and optimizer.
   - `test_accuracy`: Evaluates the model's accuracy and average loss on the test data.
- **K-fold Cross-Validation:** `K_fold_validation` performs K-fold cross-validation to tune hyperparameters and select the best model.
- **Visualization:** `plot_image_and_label` and `display_filters` are used to visualize images, labels, and filter weights.

## Experimentation

- **Hyperparameter Tuning:** The code includes K-fold cross-validation for hyperparameter tuning. Experiment with different hyperparameter ranges to achieve higher accuracy.
- **Model Architecture:** Modify the `MyMLP` and `MyCNN` classes to experiment with different architectures and layer configurations.

## Acknowledgments

- This project utilizes the PyTorch library for deep learning.
- The MNIST dataset is used for image classification.

---

**Note:** Feel free to add your name, contact information, or additional project-specific details to the `readme.md`.

Let me know if you have any other requests or modifications for the `readme.md`! 
