# CIFAR10 CNN Classification

This project involves training a Convolutional Neural Network (CNN) to perform binary classification on the CIFAR10 dataset. The classification task is to categorize objects into two groups:

- **Can fly** (class 1): Includes `airplane` and `bird`.
- **Cannot fly** (class 0): Includes `automobile`, `cat`, `deer`, `dog`, `frog`, `horse`, `ship`, `truck`.

## Dataset

The CIFAR10 dataset is a widely used dataset for machine learning and computer vision research. It consists of 60,000 32x32 color images in 10 classes, with 6,000 images per class. In this project, we only use the images belonging to the classes listed above.

## Model

We use a Convolutional Neural Network (CNN) architecture to classify the images. The model consists of the following layers:

- Convolutional layer with 32 filters, kernel size 3x3, ReLU activation
- MaxPooling layer with pool size 2x2
- Batch Normalization layer
- Dropout layer with rate 0.3

- Convolutional layer with 64 filters, kernel size 3x3, ReLU activation
- MaxPooling layer with pool size 2x2
- Batch Normalization layer
- Dropout layer with rate 0.3

- Flatten layer
- Dense layer with 128 units, ReLU activation
- Batch Normalization layer
- Dropout layer with rate 0.3

- Dense output layer with 1 unit, sigmoid activation

## Training

The model is trained using the Adam optimizer and binary crossentropy loss. We train the model for 15 epochs with a batch size of 64 and a validation split of 15%.

## Evaluation

The model is evaluated on a separate test set, and the following metrics are reported:
- Precision
- Recall
- F1 Score
- Accuracy

A confusion matrix is also generated to visualize the performance.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
