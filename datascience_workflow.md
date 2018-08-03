---
title: Data science workflow
output:
  html_document:
    toc: true
    toc_depth: 3
    toc_float: true
    number_sections: true
---
# Data science workflow

This is intended to be a template to give guidelines when exploring a new data set.

## Load packages
### The basics
```{python}
import pandas as pd
import numpy as np

from matplotlib import pyplot as plt
import seaborn as sns
```

### Preprocessing
Various preprocessing methods are available, StandardScaler is a classic one.
```{python}
from sklearn.preprocessing import StandardScaler
```

### Models
Depending on the problem, various types of models can be loaded. For instance, for classification tasks:

```{python}
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
```

### Pipeline
Using pipeline ensures avoiding data leakage when performing cross-validation.
```{python}
from sklearn.pipeline import Pipeline
```

### Model selection
Depending on the circumstances, this step requires the use of different libraries.

```{python}
from sklearn.model_selection import KFold
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score, GridSearchCV
from sklearn.metrics import mean_squared_error
from sklearn.metrics import roc_curve, auc, precision_score, confusion_matrix, make_scorer, accuracy_score
```

## Load data

## Summarize data
* Check shape of the data
* Check head and tail of the dataset for import errors, premature end of file, etc
* Show and enhanced description of the dataset, including number of distinct values per variable, number of NA, correlation with target, etc...
```{python}
target = target_col_name
stat_df = df.describe().T
stat_df = stat_df.reset_index().rename(columns={'index': 'column'})
stat_dfstat_df[['distinct_vals''distinct ] = stat_df['column'].apply(lambda x: len(df[x].value_counts()))
stat_df['target_corr'] = stat_df['column'].apply(lambda x: np.corrcoef(df[target], df[x], rowvar=False)[0][1])
stat_df['nb_NA'] = stat_df['column'].apply(lambda x: np.sum(df[x].isnull().any()))
```

### Handle special types of data
Dates
Categorical data
Missing values

### Low variance
Remove data with low variance, they don't bring information to the problem

### Overview of correlations
* Look for correlation between the features
* Look for features correlated with the output

## Evaluate algorithm
### Choose a metric to optimize
Depends on the task

### Assess performance of baseline algorithm
For classification, predict the most frequent class, for regression predict the average value

### Spot check a bunch of algorithms
Crete a list of estimators to test and evaluate their performance.


```
pandoc -o datascience_template.html datascience_template.md --template=GitHub.html5
```
