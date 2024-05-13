# Fuel efficiency prediction
Demonstration of the development and analysis of different regression models to predict fuel consumption using various car attributes from the [Auto MPG dataset](https://www.kaggle.com/datasets/uciml/autompg-dataset). An answer to a question - what decides how far a car will go? 

A full, detailed report of the process in Polish can be read in 
<a href="report.pdf">report.pdf</a>

## Features
The project includes data preprocessing, exploratory data analysis, and the implementation of a linear regression model using the following features:
<p align="center"><img width="340" src="https://github.com/NakerTheFirst/Fuel-efficiency-prediction/blob/main/coefficient_values.png" alt="Image of a table of coefficient values for LSM-based linear regression"></p>
<p align="center"><a href="report.pdf">Table 1.<alt="Report link"></a> Feature coefficient values for LSM-based linear regression</p>

## Functionalities
### Data Preprocessing
- Reads car data from a file.
- Converts MPG (Miles per Gallon) to liters per 100 km.
- Normalises selected numerical features.
- Feature engineers categorical origin variable into Europe, USA or Asia via one-hot encoding. 

### Visualisation
- Provides scatter plots and histograms to explore data relationships and distribution.
- Offers correlation heatmaps to understand feature interactions.

### Model Training and Evaluation
- Splits the data into training and testing sets.
- Trains a linear regression model and evaluates its performance using mean squared error (MSE).
- Compares the accuracy of different regression methods: LSM, Lasso, Ridge, ElasticNet.
- Outputs the parameter values for optimal LSM-based linear regression method.
<br>

<p align="center"><img width="698" src="https://github.com/NakerTheFirst/Fuel-efficiency-prediction/blob/main/evaluation_metrics.png" alt="Image of a table of MSE error evaluation for different regression methods"></p>
<p align="center"><a href="report.pdf">Table 2.<alt="Report link"></a> Minimum Squared Error, intercept and coefficients comparison for different regression types</p>

## Installation
To run this project, you need Python installed along with the following libraries:
- NumPy
- pandas
- matplotlib
- seaborn
- scikit-learn

You can install these packages using pip:
```bash
pip install numpy pandas matplotlib seaborn scikit-learn
```

## Usage
To use this project:

1. Ensure you have the auto-mpg.data file in your working directory. This file contains the dataset used for modeling.
2. Run the script via the command line:

```bash
python main.py
```
