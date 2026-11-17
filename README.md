# ft_linear_regression

An introduction to Machine Learning through a univariate linear regression model implemented from scratch in Python for **École 42**. The goal is to predict the price of a car based on its mileage using gradient descent.

## Overview

The project is split into three modular files:
* **`model.py`**: Contains the `LinearRegression` class handling dataset loading, Min-Max normalization, gradient descent with simultaneous parameter updates, denormalization, JSON persistence, $R^2$ accuracy evaluation, and data visualization.
* **`train.py`**: Trains the model on `data.csv`, saves the resulting parameters ($\theta_0$ and $\theta_1$) into `trained.json`, prints the model precision, and displays the regression plot.
* **`predict.py`**: Loads the trained parameters from `trained.json` (defaulting to `0.0` if the model has not been trained yet) and estimates the car price for a user-provided mileage.

## Requirements

* **Python 3.10+**
* **`matplotlib`** (used solely for plotting the bonus graph)

Install the visualization dependency:
```bash
pip install matplotlib
```

## Usage

1. **Before training** (returns `0.0` as `trained.json` does not exist yet):
```bash
python3 predict.py
```

2. **Train the model** (generates `trained.json`, displays the $R^2$ score, and opens the regression chart):
```bash
python3 train.py
```

3. **Predict a price** using the trained parameters:
```bash
python3 predict.py
```

## Mathematical Implementation

* **Hypothesis**:
  $$\text{estimatePrice}(\text{mileage}) = \theta_0 + \theta_1 \times \text{mileage}$$

* **Normalization & Denormalization**:
  Both mileage and price values are scaled to $[0, 1]$ using Min-Max normalization prior to gradient descent to prevent numerical overflow and ensure fast convergence (`learning_rate = 0.1`, `iterations = 3000`). Once trained, $\theta_0$ and $\theta_1$ are denormalized back to real euro/kilometer units before serialization.

* **Bonus Features**:
  * **Visualization**: Plots the dataset scatter points alongside the linear regression line using `matplotlib`.
  * **Precision Metric ($R^2$)**: Computes the coefficient of determination ($R^2 \approx 73.30\%$ on `data.csv`) to evaluate goodness of fit.