11/14/15

Tried Lasso with min_df = 120.

Did CV with alpha = 10^range(-4, 2).

As alpha decreases, convergence became really slow so tried Flux.

Result of best Lasso (alpha = 0.0001) is worse than Ridge.

Plot for cross-validation of Lasso
![alt lasso_cv](/cv plot/lasso_cv.png)
