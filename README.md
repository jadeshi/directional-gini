# directional-gini
A slightly modified version of Gini importance ("Directional Gini importance") that includes direction of correlation information between feature and model prediction

This is a slightly modified version of the Gini importance metric for measuring feature importance in decision tree regression models. Instead of giving a strictly positive value, the sign of the value for each feature will indicate whether the feature is overall positively or negatively correlated with the model prediction, analogous to beta coefficients in linear regression. 

The two functions included in the directional_gini.py file, compute_directional_gini and compute_ensemble_directional_gini, are meant to be used with scikit-learn's DecisionTreeRegressor and RandomForestRegressor objects as arguments respectively. They will return an array of length N (N being the number of total features in your input data) of directional Gini feature importance for either a single tree or a random forest.

For ideal, noiseless features with perfect positive or negative correlations with the output, the magnitude of Directional Gini is exactly equal to either the postive or negative of normal Gini importance, depending on the direction of correlation. For realistic data with noise present, the magnitude of Directional Gini will almost always be less than that of Gini.
