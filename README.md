# Simple_Linear_Regression

## Boston Housing Price Prediction using Linear Regression
This project demonstrates how to perform simple linear regression on the Boston Housing dataset to predict the median value of homes based on the average number of rooms per dwelling. The implementation uses both a custom gradient descent method and the Scikit-learn LinearRegression model for comparison.

## Table of Contents
* Overview
* Dataset
* Dependencies
* Code Explanation
* How to Run
* Results

## Overview
The goal of this project is to:

* Implement linear regression using gradient descent to optimize the model parameters.
* Compare the results with the Scikit-learn implementation of linear regression.
* Visualize the training process, the final fitted line, and the performance comparison between the two models.

## Dataset
The dataset used in this project is the Boston Housing dataset, which contains information collected by the U.S Census Service concerning housing in the area of Boston, Massachusetts. The specific columns used in this project are:

* rm: Average number of rooms per dwelling (input feature).
* medv: Median value of owner-occupied homes in $1000s (target).

## Dependencies
To run this code, you will need the following Python libraries:

* numpy
* pandas
* matplotlib
* scikit-learn

## You can install the required libraries using:

* bash
* Copy code
* pip install numpy pandas matplotlib scikit-learn

## Code Explanation

### Loading the Dataset
The dataset is loaded using pandas from a CSV file. The input feature (rm) and target variable (medv) are extracted and converted to numpy arrays for further processing.

### Gradient Descent Implementation
* compute_cost: Calculates the cost function (mean squared error) for the current values of weights (w) and bias (b).
* compute_gradient: Computes the gradients of the cost function with respect to w and b.
* gradient_descent: Iteratively updates the parameters (w and b) using the gradients until convergence is reached or the maximum number of iterations is achieved. The learning rate (alpha) controls the size of the updates.

### Model Training
* Initialization: The weights and bias are initialized to zero.
* Training: Gradient descent is run for a set number of iterations, and the cost is minimized.
* Comparison: The final model parameters are compared with those obtained using Scikit-learn's LinearRegression.

### Visualization
* Cost History: The cost function history is plotted to visualize the convergence of the gradient descent algorithm.
* Fitted Line: The trained model is visualized by plotting the input feature against the target variable and the fitted line.
* Model Comparison: The predictions of the custom gradient descent model are compared with those of the Scikit-learn model by plotting true values against predicted values.

## How to Run
* Ensure Dependencies are Installed: Make sure you have all the required Python libraries installed.
* Prepare the Dataset: Ensure the dataset file (Boston.csv) is placed in the correct path as specified in the code.
* Run the Script: Execute the script to train the model and generate the plots.

## Results
The script will output:

* The optimized weight (w) and intercept (b) for both the gradient descent and Scikit-learn models.
* The R² score for the Scikit-learn model.
* Plots showing:
* The cost function history during gradient descent.
* The linear fit of the model.
* A comparison of the predictions by the custom gradient descent model and Scikit-learn model against the true values.
