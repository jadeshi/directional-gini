# directional-gini
### Introduction
This is a slightly modified version of the Gini importance metric (Breiman L: Random forests. J Mach Learn 2001, 45: 5–32) for measuring feature importance in decision tree-based regression models. Instead of giving a strictly positive value, the sign of the value for each feature will indicate whether the feature is overall positively or negatively correlated with the independent variable, analogous to beta coefficients in linear regression. 

This code is a modified version of a function for calculating normal Gini importance located at the following link: 
https://stackoverflow.com/questions/49170296/scikit-learn-feature-importance-calculation-in-decision-trees


The two functions included in the directional_gini.py file, compute_directional_gini and compute_ensemble_directional_gini, are meant to be used with scikit-learn's (https://scikit-learn.org/stable/) DecisionTreeRegressor and RandomForestRegressor objects as arguments respectively. They will return an array of length N (N being the number of total features in your input data) of directional Gini feature importance for either a single tree or a random forest.


For ideal, noiseless data with perfect positive or negative correlation between feature and independent variable, the magnitude of Directional Gini is exactly equal to either the postive or negative of normal Gini importance, depending on the direction of correlation. For realistic data with noise present, the magnitude of Directional Gini will almost always be less than that of Gini.

### Example Usage
```
from sklearn.tree import DecisionTreeRegressor
import numpy as np
from directional_gini import *

X = np.array([np.arange(10)]).T
X = np.hstack((X, X * -1))
y = np.arange(10)

dt = DecisionTreeRegressor()
dt.fit(X,y)

# Regular Gini importances
dt.feature_importances_ # array([0.1030303, 0.8969697])

# Directional Gini importances
compute_directional_gini(dt) # array([ 0.1030303, -0.8969697])
```


