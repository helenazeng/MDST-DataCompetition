#
# MDST Ratings Analysis Challenge
# Lasso with Cross Validation
#
# Prerequisites:
#
# numpy
# pandas
# sklearn
#

import pandas as pd
import numpy as np

from sklearn import linear_model, cross_validation
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS, CountVectorizer
from sklearn.metrics import mean_squared_error

np.random.seed(0)

# Load in the data - pandas DataFrame objects
rats_tr = pd.read_csv('data/train.csv')
rats_te = pd.read_csv('data/test.csv')

# Construct bigram representation
count_vect = CountVectorizer(min_df=20,ngram_range=(1,2))

# "Fit" the transformation on the training set and apply to test
Xtrain = count_vect.fit_transform(rats_tr.comments.fillna(''))
Xtest = count_vect.transform(rats_te.comments.fillna(''))

Ytrain = np.ravel(rats_tr.quality)

# # Select alpha with a validation set
# Xtr, Xval, Ytr, Yval = cross_validation.train_test_split(
#     Xtrain,
#     Ytrain,
#     test_size = 0.3,
#     random_state = 0)
#
# # Define window to search for alpha
# alphas = np.power(10.0, np.arange(-4, 2))
#
# # Store rmses here for plotting
# rmseTr = np.zeros((len(alphas),))
# rmseVal = np.zeros((len(alphas),))
#
# # Search for lowest validation accuracy
# for i in range(len(alphas)):
#     print "alpha =", alphas[i]
#     m = linear_model.Lasso(alpha = alphas[i])
#     m.fit(Xtr, Ytr)
#     YhatTr = m.predict(Xtr)
#     YhatVal = m.predict(Xval)
#     rmseTr[i] = np.sqrt(mean_squared_error(YhatTr, Ytr))
#     rmseVal[i] = np.sqrt(mean_squared_error(YhatVal, Yval))
#     print alphas[i], rmseTr[i], rmseVal[i]
#
# import matplotlib
# matplotlib.use('Agg')
# import matplotlib.pyplot as plt
# plt.semilogx(alphas, rmseTr, hold=True)
# plt.semilogx(alphas, rmseVal)
# plt.legend(['Training RMSE', 'Validation RMSE'])
# plt.ylabel('RMSE')
# plt.xlabel('alpha')
# plt.draw()
# plt.savefig('lasso_cv.png')

# Best performance at alpha = 0.0001
# Train new model using all of the training data
m = linear_model.Ridge(alpha = 0.0001)
m.fit(Xtrain, Ytrain)
Yhat = m.predict(Xtest)

# Save results in kaggle format
submit = pd.DataFrame(data={'id': rats_te.id, 'quality': Yhat})
submit.to_csv('lasso_submit.csv', index = False)

# Other things to try:
# Add other features
# Other regularization types (lasso)
#     http://scikit-learn.org/stable/modules/generated/sklearn.linear_model.Lasso.html
# Decision trees
#     http://scikit-learn.org/stable/modules/generated/sklearn.tree.DecisionTreeRegressor.html
# Random forests
#     http://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestRegressor.html
# Boosting
#     http://scikit-learn.org/stable/modules/generated/sklearn.ensemble.GradientBoostingRegressor.html
