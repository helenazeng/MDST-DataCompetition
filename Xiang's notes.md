11/14/15

Tried Lasso with min_df = 120.

As alpha decreases, convergence became really slow so tried Flux.

Result of best Lasso (alpha = 0.0001) is worse than Ridge.

Plot for cross-validation of Lasso
![alt lasso_cv](/cross-validation plot/lasso_cv.png)
Convergence is extremely slow for alpha < 0.0001

Plot for cross-validation of Ridge using all features
![alt lasso_cv](/cross-validation plot/ridge_all_cv.png)

Plot for cross-validation of Ridge using all features
![alt lasso_cv](/cross-validation plot/tree_all_cv.png)


11/22/15
Xgboost gave a better result then ridge regression. Current best result is from

min_df = 20

max_depth = 6

eta = 0.2

num_round = 750

Changing min_df from 20 to 10 gave negligible improvement
