This repository contains quick data science studies performed on publicly available datasets.

# Notebook contents
[pima_indians_diabetes.ipynb](pima_indians_diabetes.ipynb): this notebook demonstrates a quick Exploratory Data Analysis as well as modeling on a classification problem.

The main interests are:
* an enhancement of Python's `describe()` command, so that it displays number of NAs, distinct values and correlation with target, for each variable
* use of Seaborn for plots, whch leads to effortless display of correlation matrix, violin plots and scatterplot matrix
* use of `sklearn.pipeline` module, to create a robust data processing pipeline and avoid data leakage
* grid search optimization and feature importance display for Random Forest classifier
