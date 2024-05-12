# Parkinsons Telemonitoring Insights

[![Contributions Welcome](https://img.shields.io/badge/contributions-welcome-brightgreen.svg?style=flat&label=Contributions&colorA=red&colorB=black)](#)
[![Visitors](https://visitor-badge.laobi.icu/badge?page_id=AtharvaKulkarniIT.ParkinsonsTelemonitoringInsights)](https://github.com/AtharvaKulkarniIT/ParkinsonsTelemonitoringInsights)
[![LOC](https://sloc.xyz/github/AtharvaKulkarniIT/ParkinsonsTelemonitoringInsights)](https://github.com/AtharvaKulkarniIT/ParkinsonsTelemonitoringInsights)

## Overview

This data science project in R aims to predict the severity of Parkinson's disease based on the UCI Parkinsons dataset using machine learning algorithms. The dataset includes various features related to Parkinson's symptoms, and we leverage decision tree, random forest, support vector machine (SVM) and XGBoost algorithms for prediction. Additionally, Lasso regularization is applied for feature selection to enhance model interpretability and efficiency.

## Table of Contents

- [Dataset](#dataset)
- [Preprocessing](#preprocessing)
- [Feature Selection](#feature-selection)
- [Models](#models)
  - [Decision Tree](#decision-tree)
  - [Random Forest](#random-forest)
  - [Support Vector Machine (SVM)](#support-vector-machine-svm)
  - [XGBoost](#xgboost)
- [Hyperparameter Tuning](#hyperparameter-tuning)
- [Evaluation](#evaluation)
- [Shiny App](#shiny-app)
- [Usage](#usage)
- [Report](#report)
- [Contributing](#contributing)
- [License](#license)

## Dataset

We use the [UCI Parkinsons dataset](https://archive.ics.uci.edu/dataset/189/parkinsons+telemonitoring) for this data science project. The dataset includes information about various symptoms and features related to Parkinson's disease.Motor UPDRS and Total UPDRS are target variables in this dataset.

## Preprocessing

The data is preprocessed by scaling the features using min-max scaling to ensure uniformity and enhance model performance.

## Feature Selection

Lasso regularization is applied for feature selection to identify the most relevant features, enhancing model interpretability and efficiency.

## Models

### Decision Tree

We employ a decision tree regression model to predict the severity of Parkinson's disease based on the dataset features.

### Random Forest

A random forest regression model is utilized for predicting the severity of Parkinson's disease, offering an ensemble approach for improved accuracy.

### Support Vector Machine (SVM)

The support vector machine is employed for regression to predict the severity of Parkinson's disease.

### XGBoost

XGBoost, an efficient gradient boosting algorithm, is used to predict disease severity, providing a robust alternative to traditional models.

## Hyperparameter Tuning

To optimize model performance, hyperparameter tuning is performed for each algorithm.

## Evaluation

The performance of each model is evaluated using metrics such as RMSE (Root Mean Squared Error), R-squared and MAE (Mean Absolute Error).

## Shiny App

A Shiny app is developed to provide an interactive interface for visualizing and analyzing the predictions made by the random forest model.

## Usage

1. Clone the repository.
2. Ensure all project files are in the same folder.
3. Open R Studio and set the working directory to the project folder.
4. Install the required R libraries using the following command:
   ```R
   install.packages(c("dplyr", "e1071", "rpart", "randomForest", "caTools", "corrplot", "xgboost", "Hmisc", "caret", "glmnet"))
   ```
5. Run the following R scripts in order:
   - `Data_Preprocessing.R`: Preprocess the data and apply min-max scaling.
   - `Decision_Tree.R`: Run the decision tree regression model.
   - `RandomForest.R`: Train the random forest regression model and save .
   - `SVM.R`: Employ the support vector machine for regression.
   - `XGBoost.R`: Utilize XGBoost for predicting disease severity.
6. Run the Shiny app by executing `app.R`. Ensure that the RF trained models files are in the same directory as the app.

## Report

For a complete report or further inquiries, feel free to contact us via email [click here](mailto:atharva9412@gmail.com).

## Contributing

Contributions are welcome! Feel free to open issues or pull requests.

## License

This project is licensed under the [License](LICENSE). Please read carefully.

