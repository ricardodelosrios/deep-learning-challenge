# deep-learning-challenge

<div style="display: inline_block"><br/>
  <img align="center" alt="Colaboratory" src="https://img.shields.io/badge/Colab-F9AB00?style=for-the-badge&logo=googlecolab&color=525252" />

<div style="display: inline_block"><br/>
  <img align="center" alt="python" src="http://ForTheBadge.com/images/badges/made-with-python.svg" />


## Introduction

Charity Success Prediction is a machine learning project that predicts the likelihood of a charity organization's success based on various features. This project employs deep neural networks and advanced hyperparameter tuning techniques to achieve accurate predictions.

## Libraries Used

The project utilizes several libraries to accomplish its tasks:

- `sklearn.model_selection`: This library provides functions for splitting the data into training and testing sets, a crucial step in machine learning model development.

- `sklearn.preprocessing`: The `StandardScaler` from this library is used to standardize the input features, making the training process more efficient.

- `pandas`: Pandas is used for data manipulation, including reading the dataset, data preprocessing, and creating dummy variables for categorical data.

- `tensorflow`: TensorFlow is a powerful deep learning framework used to build and train the neural network model.

- `tensorflow.keras.callbacks`: The `EarlyStopping` callback is used to prevent overfitting and reduce training time.

- `keras_tuner`: This library is employed for hyperparameter tuning, enabling the optimization of the model's architecture.

## Project Description

The project performs the following tasks:

1. Data Loading: It starts by loading a dataset from a remote source, the "charity_data.csv."

2. Data Preprocessing: The project preprocesses the data by removing non-beneficial ID columns ('EIN' and 'NAME'). It also handles categorical data by applying binning to reduce the number of unique values in the 'APPLICATION_TYPE' and 'CLASSIFICATION' columns. The categorical data is further converted to numeric format using one-hot encoding.

3. Data Splitting: The preprocessed data is split into training and testing datasets.

4. Model Building: A deep neural network model is constructed with multiple hidden layers. The model architecture is designed to learn patterns from the input features and make predictions on the likelihood of charity success.

5. Model Training: The model is compiled, and the training process begins. An early stopping mechanism is applied to prevent overfitting. The training process is logged, and the model's performance is evaluated.

6. Model Evaluation: The trained model is evaluated using the test data to calculate loss and accuracy.

7. Model Export: The final trained model is saved in an HDF5 file, 'Charity_Model.h5,' for future use.

## Functionality

To understand how the provided code works, let's break it down step by step:

### Library Imports:

The code begins by importing several Python libraries, including keras-tuner, scikit-learn, pandas, and tensorflow. These libraries are essential for various tasks within the project.

### Data Loading and Initial Inspection:

The dataset is loaded from an online source using Pandas' read_csv function. The dataset is stored in the application_df variable.
Initial data inspection is performed by displaying the first few rows of the dataset using application_df.head(). This helps in understanding the structure and content of the data.

### Data Preprocessing:

The code proceeds to preprocess the dataset. Non-beneficial columns, 'EIN' and 'NAME', are dropped using the drop method. These columns are not relevant for predicting the success of charitable applications.
A crucial part of data preprocessing is handling categorical variables. In this code, categorical data in the 'APPLICATION_TYPE' and 'CLASSIFICATION' columns is binned to handle categories with a low count. The low-count categories are grouped into an 'Other' category. This simplifies the dataset and reduces the dimensionality, making it more manageable for model training.
### Feature Engineering:

Categorical data is converted into a numeric format using one-hot encoding. The pd.get_dummies function is applied to create binary columns for each category, indicating the presence or absence of each category in a data point. This process ensures that the categorical data can be used as input for the neural network model.
The dataset is then split into two components: feature arrays (X) and target arrays (y). X represents the features used for prediction, and y represents the target variable to predict, which is whether an applicant will be successful if funded by Alphabet Soup.

### Data Standardization:

Standardization is an important step to ensure that all features are on the same scale. The StandardScaler from scikit-learn is used to standardize the feature data.
Standardization brings all numerical features to a common mean and standard deviation, preventing features with larger numerical values from dominating the training process.

### Model Creation:

A deep neural network model is created using TensorFlow and Keras. The model architecture is defined step by step.
The model starts with an input layer with 80 units and a ReLU activation function. The choice of architecture, including the number of units, is a key decision that can be fine-tuned to optimize model performance.
A second hidden layer with 30 units and a sigmoid activation function follows.
The output layer consists of a single unit with a sigmoid activation function, suitable for binary classification tasks.

### Model Compilation:

The model is compiled with the necessary configuration for training. The loss function is set to "binary_crossentropy" since this is a binary classification task. The optimization algorithm is "Nadam," a variation of stochastic gradient descent (SGD).
Metrics, in this case, "accuracy," are specified to evaluate the model's performance.
### Model Training:

The model is trained on the preprocessed and standardized training data (X_train_scaled and y_train) for a specified number of epochs (500 in this code).
An early stopping callback is implemented with EarlyStopping to monitor the training process. It stops training if the loss stops improving and restores the best weights to prevent overfitting.
### Model Evaluation:

The trained model is evaluated using the test data (X_test_scaled and y_test) to assess its performance. The model's loss and accuracy on the test data are computed.
This step helps determine how well the model generalizes to unseen data.

### Model Export:

Finally, the trained model is saved in an HDF5 file format using TensorFlow's nn.save function. This saved model can be used for future predictions without the need to retrain the model.

## How to Use

To use this project, follow these steps:

1. Install the required libraries:

   ```bash
   pip install keras-tuner scikit-learn pandas tensorflow


## Developer

[<img src="https://avatars.githubusercontent.com/u/133066908?v=4" width=115><br><sub>Ricardo De Los Rios</sub>](https://github.com/ricardodelosrios) 
