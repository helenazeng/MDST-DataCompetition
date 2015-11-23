#
# MDST Ratings Analysis Challenge
# Model selection with Cross Validation
#
# Jonathan Stroud
#
#
# Prerequisites:
#
# numpy
# pandas
# sklearn
#
import pdb

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

# Select alpha with a validation set
Xtr, Xval, Ytr, Yval = cross_validation.train_test_split(
    Xtrain,
    Ytrain,
    test_size = 0.3,
    random_state = 0)

# Define window to search for alpha
alphas = np.power(10.0, np.arange(-2, 8))

# Store MSEs here for plotting
mseTr = np.zeros((len(alphas),))
mseVal = np.zeros((len(alphas),))

# Search for lowest validation accuracy
for i in range(len(alphas)):
    print "alpha =", alphas[i]
    m = linear_model.Ridge(alpha = alphas[i])
    m.fit(Xtr, Ytr)
    YhatTr = m.predict(Xtr)
    YhatVal = m.predict(Xval)
    mseTr[i] = mean_squared_error(YhatTr, Ytr)
    mseVal[i] = mean_squared_error(YhatVal, Yval)
    print alphas[i], mseTr[i], mseVal[i]

import matplotlib.pyplot as plt
plt.semilogx(alphas, mseTr, hold=True)
plt.semilogx(alphas, mseVal)
plt.legend(['Training MSE', 'Validation MSE'])
plt.ylabel('MSE')
plt.xlabel('alpha')
plt.show()

# Best performance at alpha = 100
# Train new model using all of the training data
m = linear_model.Ridge(alpha = 100)
m.fit(Xtrain, Ytrain)
Yhat = m.predict(Xtest)

# Save results in kaggle format
submit = pd.DataFrame(data={'id': rats_te.id, 'quality': Yhat})
submit.to_csv('crossvalidation_submit.csv', index = False)

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
