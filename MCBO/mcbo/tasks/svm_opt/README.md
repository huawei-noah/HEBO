# SVM hyperparameter tuning

Given `slice localisation` dataset loaded from `UCI`, the task consist in optimising hyperparameters of
`SVR` regressor from `sklearn and dataset features:

- gamma: \[1e-1, 10\] (search in `pow`)
- C: \[1e-2, 100\] (search in `pow`)
- epsilon: \[1e-2, 1\]  (search in `pow`)
- (for i=1,...,50) feature_i: {0, 1}, whether to select feature `i` to fit the model. 

This is a mixed search space with 50 boolean dimensions and 3 numerical (power scale) dimensions.

The blackbox is the RMSE obtained over 5 cross-validation splits, when fitting `SVR` with provided hyperparameters and
 considering only the selected features.

Slice localisation dataset is pre-processed: start by removing the features that are constant in the dataset. Sample 
10,000 random points, run XGBoost to order the features by their predictive power, and keep the 50 most predictive.

## Setup

```shell
pip install sklearn

wget https://archive.ics.uci.edu/ml/machine-learning-databases/00206/slice_localization_data.zip
unzip slice_localization_data.zip
rm slice_localization_data.zip
mv  slice_localization_data.csv ../data/
```

## Acknowledgements

This task has been considered in
[Bayesian Optimization over High-Dimensional Combinatorial Spaces via Dictionary-based Embeddings]
(https://arxiv.org/pdf/2303.01774), Deshwal et al. 2023.
