
# Parkinsons Telemonitoring Insights
<a href="https://github.com/AtharvaKulkarniIT/ParkinsonsTelemonitoringInsights"><img src="https://sloc.xyz/github/AtharvaKulkarniIT/ParkinsonsTelemonitoringInsights" alt="LOC"/></a>
<a href="https://github.com/AtharvaKulkarniIT/ParkinsonsTelemonitoringInsights"><img src="https://visitor-badge.laobi.icu/badge?page_id=AtharvaKulkarniIT.ParkinsonsTelemonitoringInsights" alt="Visitors"/></a>
<a href="https://github.com/AtharvaKulkarniIT/ParkinsonsTelemonitoringInsights/stargazers"><img src="https://img.shields.io/github/stars/AtharvaKulkarniIT/ParkinsonsTelemonitoringInsights" alt="Stars Badge"/></a>
[![GitHub Forks](https://img.shields.io/github/forks/AtharvaKulkarniIT/ParkinsonsTelemonitoringInsights.svg?style=social&label=Forks&maxAge=2592000)](https://www.github.com/AtharvaKulkarniIT/ParkinsonsTelemonitoringInsights/fork)
[![contributors ](https://img.shields.io/badge/contributions-welcome-brightgreen.svg?style=flat&label=Contributions&colorA=red&colorB=black	)](#)


This data science project in R aims to predict the severity of Parkinson's disease based on the UCI Parkinsons dataset using machine learning algorithms. The dataset includes various features related to Parkinson's symptoms, and we leverage decision tree, random forest, support vector machine (SVM) and XGBoost algorithms for prediction. The project also involves hyperparameter tuning and feature selection to enhance model performance.

## Table of Contents

- [Dataset](#dataset)
- [Preprocessing](#preprocessing)
- [Models](#models)
  - [Decision Tree](#decision-tree)
  - [Random Forest](#random-forest)
  - [Support Vector Machine (SVM)](#support-vector-machine-svm)
  - [XGBoost](#xgboost)
- [Hyperparameter Tuning](#hyperparameter-tuning)
- [Feature Selection](#feature-selection)
- [Evaluation](#evaluation)
- [Correlation Analysis](#correlation-analysis)
- [Usage](#usage)
- [Contributing](#contributing)
- [License](#license)

## Dataset

We use the [UCI Parkinsons dataset](https://archive.ics.uci.edu/dataset/189/parkinsons+telemonitoring) for this data science project. The dataset includes information about various symptoms and features related to Parkinson's disease.

## Preprocessing

The data is preprocessed by scaling the features using min-max scaling to ensure uniformity and enhance model performance.

## Models

### Decision Tree

We employ a decision tree regression model to predict the total UPDRS score based on the dataset features.

### Random Forest

A random forest regression model is utilized for predicting the severity of Parkinson's disease, offering an ensemble approach for improved accuracy.

### Support Vector Machine (SVM)

The support vector machine is employed for regression to predict the total UPDRS score.

### XGBoost

XGBoost, an efficient gradient boosting algorithm, is used to predict disease severity, providing a robust alternative to traditional models.

## Hyperparameter Tuning

To optimize model performance, hyperparameter tuning is performed for each algorithm.

## Feature Selection

Feature selection techniques are applied to identify the most relevant features, enhancing model interpretability and efficiency.

## Evaluation

The performance of each model is evaluated using metrics such as RMSE (Root Mean Squared Error), R-squared, and MAE (Mean Absolute Error).

## Correlation Analysis

A correlation heatmap is generated to explore relationships between different features in the dataset.

## Usage

1. Clone the repository.
2. Install the required R libraries using the `install.packages` command.
3. Run the R script to preprocess the data, train the models, and evaluate their performance.

```R
parkinsons_analysis.R
```

## Contributing

Contributions are welcome! Feel free to open issues or pull requests.

## License

This data science project in R is licensed under the [MIT License](LICENSE).

---

