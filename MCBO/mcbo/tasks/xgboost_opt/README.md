# XG-BOOST hyperparameter tuning

Given a dataset loaded from `sklearn` library (e.g. MNIST, Boston...) the task consist in optimising hyperparameters of
an XG-Boost classifier / regressor:

- booster type: {gbtree, dart}
- grow policy: {depthwise, loss}
- training objective: {softmax, softprob}
- learning rate: \[1e-5, 1\]  (search in `pow`)
- max depth: \[1, 10\]  (search in `int`)
- minimum split loss: \[0, 10\]
- subsample: \[0.001, 1\]
- amount of regularisation: \[0, 5\]

This is a mixed search space with 5 numerical and 3 categorical dimensions.


## Setup

```shell
pip install xgboost, sklearn
```

**Remark:** we observed that running this task on a small number of cores is actually faster than when many cores are
used.

## Acknowledgements

This task has been initially presented in
[Bayesian optimization over multiple continuous and categorical inputs](https://www.google.com/url?sa=t&rct=j&q=&esrc=s&source=web&cd=&cad=rja&uact=8&ved=2ahUKEwiHqZ237ZP6AhU1_7sIHdCBA18QFnoECA4QAQ&url=https%3A%2F%2Farxiv.org%2Fabs%2F1906.08878&usg=AOvVaw05WOh0_OOBRxflDKoGBqIF)
, Ru et al. 2020.

Our implementation is an adaptation of
[Casmopolitan task implementation](https://github.com/xingchenwan/Casmopolitan/blob/ae7f5a06206712e7776562c5c0e8f771c8780575/mixed_test_func/xgboost_hp.py#L117)
.

Compared to these we modified the nature of learning rate parameter to a `pow` and max depth to an 
`int`.