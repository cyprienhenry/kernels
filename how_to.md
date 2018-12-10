# Pandas DataFrames
## Rename columns

```
# remove leading and trailing space
df.columns = df.columns.str.strip().str.lower()

# replace space and slash with underscore
df.columns = df.columns.str.replace(' ', '_').str.replace('/', '_')  
```

## Show NA ratio per variable
```
df.apply(lambda x: round(x.isnull().sum()/x.shape[0]*100, 1), axis=0).sort_values(ascending=False)
```

## Convert column type to numeric
```
df[col] = pd.to_numeric(df[col])
```

## Convert column type to date
```
def convert_date(x):
  try:
    return datetime.strptime(x, '%d/%m/%Y')
  except:
    return np.nan

df[col] = df[col].apply(lambda x: conert_date(x)
```

## Impute missing values using mean
```python
df[col].fillna(df[col].mean(), inplace=True)
```

## Impute missing values using mde
```python
df[col].fillna(df[col].mode()[0], inplace=True)
```

# Plotting
## Plot correlation matrix
```python
import seaborn as sns
corr_matrix = df.corr()
plt.figure(figsize=(5, 5))
sns.heatmap(corr_matrix, cmpa=sns.diverging_palette(240, 10, as_cmap=True), square=True)
plt.xticks(rotation=80);
```

## Choose discrete colors for `hue` (scatter plot)
One way to do this is to supply a dictionary for the color palette:
`sns.scatterplot(data=df, x='x', y='y', hue='hue', palette={0:'green', 1:'red})`

# Modeling
## Useful imports
### Feature preparation
* [Binarize labels 1-vs-all](https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.label_binarize.html)
`from sklearn.preprocessing import label_binarize`
* [Standard Scaling](https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.StandardScaler.html)
`from sklearn.preprocessing import StandardScaler`
* [Label encoder](https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.LabelEncoder.html)
from sklearn.preprocessing import LabelEncoder
* [OneHotEncoder](https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.OneHotEncoder.html)
`from sklearn.preprocessing import OneHotEncoder`

* [Polynomial Features](https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.PolynomialFeatures.html)
`from sklearn.preprocessing import PolynomialFeatures`

### Feature selection

* [Recursive Feature Elimination](https://scikit-learn.org/stable/modules/generated/sklearn.feature_selection.RFE.html)
`from sklearn.feature_selection import RFE`

### Models

* [Linear Regression](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LinearRegression.html#sklearn.linear_model.LinearRegression)
`from sklearn.linear_model import LinearRegression`

* [Logistic Regression](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html)
`from sklearn.linear_model import LogisticRegression`

* [Random Forest Classifier](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html)
`from sklearn.ensemble import RandomForestClassifier`

* [Gradient Boosting Classifier](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.GradientBoostingClassifier.html)
`from sklearn.ensemble import GradientBoostingClassifier`

### Performance evaluation
* [ROC AUC Score](http://scikit-learn.org/stable/modules/generated/sklearn.metrics.roc_auc_score.html)
`from sklearn.metrics import roc_auc_score`

* [ROC Curve](http://scikit-learn.org/stable/modules/generated/sklearn.metrics.roc_curve.html)
`from sklearn.metrics import roc_curve`

* [Confusion Matrix](http://scikit-learn.org/stable/modules/generated/sklearn.metrics.confusion_matrix.html)
`from sklearn.metrics import confusion_matrix`

* [Make scorer](http://scikit-learn.org/stable/modules/generated/sklearn.metrics.make_scorer.html)
`from sklearn.metrics import make_scorer`


## Datasets creation
### Train / test sets
The classical way is:
```
# features is a Pandas DataFrame with the useful features only and target is a Pandas Series
X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.3, random_state=7, shuffle=True

# To perform CV in case of inbalanced classes
kfold= StratifiedKFold(n_splits=3, random_state=7)
```

## Model training
Example of a basic grid search:
```
model = RandomForestClassifier(class_weight='balanced', random_state=7, n_jobs=-1)
param_grid = {'n_estimators': [700, 800, 1000],
             'max_features': [5, 6, 7],
              'max_depth': [3, 4, 5]}
              
grid_search = GridSearchCV(model, 
                           param_grid, 
                           scoring='recall_macro', 
                           cv=kfold)

grid_result = grid_search.fit(X_train, y_train);
print("Best score: %.2f using %s" % (grid_result.best_score_, grid_result.best_params_))
```
## Performance analysis
### Feature importance
To display feature importance from the best model of a grid search:
```
model = grid_result.best_estimator_
importance = model.feature_importances_ 
plt.figure(figsize=(15, 10))
ft_imp = pd.DataFrame({'feature': features.columns.values, 'importance': importance}).sort_values('importance', ascending=False)
sns.barplot(data=ft_imp[:30], y='feature', x='importance')
```
