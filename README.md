This repository contains data science studies performed on publicly available datasets.


# Contents
## Pima Indian Diabetes dataset
This notebook ([link](pima_indians_diabetes.ipynb) shows a quick Exploratory Data Analysis as well as different model testing on a *classification problem*.

The main interests are:
* an enhancement of Python's `describe()` command, so that for each variable it displays the number of NAs, distinct values and correlation with target
* use of Seaborn for plots, which leads to effortless display of correlation matrix, violin plots and scatterplot matrix
* use of `sklearn.pipeline` module, to create a robust data processing pipeline and avoid data leakage when cross-validating
* grid search optimization and feature importance display for Random Forest classifier
