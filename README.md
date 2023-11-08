# deep-learning-challenge

<div style="display: inline_block"><br/>
  <img align="center" alt="Colaboratory" src="https://img.shields.io/badge/Colab-F9AB00?style=for-the-badge&logo=googlecolab&color=525252" />

<div style="display: inline_block"><br/>
  <img align="center" alt="python" src="http://ForTheBadge.com/images/badges/made-with-python.svg" />

<div style="display: inline_block"><br/>
  <img align="center" alt="Tensorflow" src="https://img.shields.io/badge/TensorFlow-FF6F00?style=for-the-badge&logo=tensorflow&logoColor=white" />



  ## Introduction

In an effort to optimize its impact on society, the non-profit Alphabet Soup Foundation seeks to develop a tool that allows it to identify funding applicants with the highest probability of success in their charitable projects.

The project focuses on predicting the success of charitable organizations, from pre-processing the data to creating and training a neural network model. In order to improve Alphabet Soup's decision making and its ability to generate a positive impact on society.

## Libraries

The project utilizes several libraries and tools, including:

`keras-tuner`: it is a powerful library used for hyperparameter tuning, specifically designed to optimize neural network models. It helps in the systematic exploration of various hyperparameter configurations to find the most effective settings for a given neural network architecture.

`scikit-learn`: It is a versatile machine learning library that provides a wide range of tools for various aspects of the machine learning workflow. In this project, it is primarily used for data preprocessing, dataset splitting, and standardization.

`pandas`: It is a popular library for data manipulation and analysis. In this project, Pandas is used for handling the dataset, including reading data from an online source, cleaning, transforming, and preparing it for training.

`tensorflow`: It  is an open-source deep learning library developed by Google. In this project, TensorFlow is used for creating, building, and training deep neural network models.

## How the Code Works

The project performs the following detailed steps:

**Library Imports**: The code begins by importing several Python libraries, including `keras-tuner`, `scikit-learn`, `pandas`, and `tensorflow`. These libraries are essential for various tasks within the project.

**Data Loading and Initial Inspection**: The dataset is loaded from an online source using Pandas' read_csv function. The dataset is stored in the application_df variable.
Initial data inspection is performed by displaying the first few rows of the dataset using application_df.head(). 

**Data Preprocessing**: The code proceeds to preprocess the dataset. Non-beneficial columns, 'EIN' and 'NAME', are dropped using the drop method. These columns are not relevant for predicting the success of charitable applications.
A crucial part of data preprocessing is handling categorical variables. In this code, categorical data in the `'APPLICATION_TYPE'` and `'CLASSIFICATION'` columns is binned to handle categories with a low count. The low-count categories are grouped into an `'Other'` category. This simplifies the dataset and reduces the dimensionality, making it more manageable for model training.

**Feature Engineering**: Categorical data is converted into a numeric format using one-hot encoding. The `pd.get_dummies` function is applied to create binary columns for each category, indicating the presence or absence of each category in a data point. This process ensures that the categorical data can be used as input for the neural network model.
The dataset is then split into two components: **feature arrays (X)** and **target arrays (y)**. X represents the features used for prediction, and y represents the target variable to predict, which is whether an applicant will be successful if funded by Alphabet Soup.

**Data Standardization**: Standardization is an important step to ensure that all features are on the same scale. The StandardScaler from scikit-learn is used to standardize the feature data.
Standardization brings all numerical features to a common mean and standard deviation, preventing features with larger numerical values from dominating the training process.

**Model Creation**: A deep neural network model is created using TensorFlow and Keras. The model architecture is defined step by step.

The model starts with an input layer with 80 units and a ReLU activation function. The choice of architecture, including the number of units, is a key decision that can be fine-tuned to optimize model performance.
A second hidden layer with 30 units and a sigmoid activation function follows.
The output layer consists of a single unit with a sigmoid activation function, suitable for binary classification tasks.

**Model Compilation**: The model is compiled with the necessary configuration for training. The loss function is set to "binary_crossentropy" since this is a binary classification task. The optimization algorithm is "Nadam," a variation of stochastic gradient descent (SGD).
Metrics, in this case, "accuracy," are specified to evaluate the model's performance.

**Model Training**: The model is trained on the preprocessed and standardized training data (X_train_scaled and y_train) for a specified number of epochs (500 in this code).
An early stopping callback is implemented with EarlyStopping to monitor the training process. It stops training if the loss stops improving and restores the best weights to prevent overfitting.

**Model Evaluation**: The trained model is evaluated using the test data (X_test_scaled and y_test) to assess its performance. The model's loss and accuracy on the test data are computed.
This step helps determine how well the model generalizes to unseen data.

**Model Export**: Finally, the trained model is saved in an HDF5 file format using TensorFlow's nn.save function. This saved model can be used for future predictions without the need to retrain the model.

## Developer

[<img src="https://avatars.githubusercontent.com/u/133066908?v=4" width=115><br><sub>Ricardo De Los Rios</sub>](https://github.com/ricardodelosrios) 
