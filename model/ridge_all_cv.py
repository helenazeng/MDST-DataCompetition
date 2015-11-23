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
from scipy.sparse import csr_matrix, hstack

from sklearn import linear_model, cross_validation
from sklearn.preprocessing import Imputer, scale
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS, CountVectorizer
from sklearn.metrics import mean_squared_error

np.random.seed(0)

# Load in the data - pandas DataFrame objects
print 'Load data'
rats_tr = pd.read_csv('data/newtrain.csv')
rats_te = pd.read_csv('data/newtest.csv')

# Construct bigram representation
print 'Construct bigram representation'
count_vect = CountVectorizer(min_df=20,ngram_range=(1,2))

# "Fit" the transformation on the training set and apply to test
Xtrain = count_vect.fit_transform(rats_tr.comments.fillna(''))
Xtest = count_vect.transform(rats_te.comments.fillna(''))

Ytrain = np.ravel(rats_tr.quality)

# Combine comments with all other features
print 'Combine all features'
for col in ['id', 'tid', 'helpfulness', 'clarity', 'easiness', 'quality', 'comments']:
    rats_tr.drop(col, axis=1, inplace=True)

TEST_ID = rats_te.id  # Preserving test data id
for col in ['id', 'tid', 'comments']:
    rats_te.drop(col, axis=1, inplace=True)

# Impute train and test data
Xtrain = hstack([csr_matrix(rats_tr.as_matrix()), Xtrain])
Xtest = hstack([csr_matrix(rats_te.as_matrix()), Xtest])
imp = Imputer(missing_values='NaN', strategy='mean', axis=0, copy=False)
Xtrain = imp.fit_transform(Xtrain)
Xtest = imp.fit_transform(Xtest)

# # Scale data to zero mean and unit variance
# Xtrain = scale(Xtrain)
# Xtest = scale(Xtest)

# Select alpha with a validation set
Xtr, Xval, Ytr, Yval = cross_validation.train_test_split(
    Xtrain,
    Ytrain,
    test_size = 0.3,
    random_state = 0)

# Define window to search for alpha
# alphas = np.power(10.0, np.arange(3, 7))
alphas = [100]

# Store MSEs here for plotting
mseTr = np.zeros((len(alphas),))
mseVal = np.zeros((len(alphas),))

# Search for lowest validation accuracy
print 'cross-validation'
for i in range(len(alphas)):
    print "alpha =", alphas[i]
    m = linear_model.Ridge(alpha = alphas[i])
    m.fit(Xtr, Ytr)
    YhatTr = m.predict(Xtr)
    YhatVal = m.predict(Xval)
    mseTr[i] = np.sqrt(mean_squared_error(YhatTr, Ytr))
    mseVal[i] = np.sqrt(mean_squared_error(YhatVal, Yval))
    print alphas[i], mseTr[i], mseVal[i]

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.semilogx(alphas, mseTr, hold=True)
plt.semilogx(alphas, mseVal)
plt.legend(['Training RMSE', 'Validation RMSE'])
plt.ylabel('RMSE')
plt.xlabel('alpha')
plt.draw()
plt.savefig('ridge_all_cv.png')

# Best performance at alpha = 100
# Train new model using all of the training data
print 'Fit all training data'
m = linear_model.Ridge(alpha = 100)
m.fit(Xtrain, Ytrain)
Yhat = m.predict(Xtest)

# Save results in kaggle format
submit = pd.DataFrame(data={'id': TEST_ID, 'quality': Yhat})
submit.to_csv('ridge_all_cv_submit.csv', index = False)


# Peak the ridge regression model
df = pd.DataFrame(data={'words':np.concatenate([rats_tr.columns, count_vect.get_feature_names()]),
                        'coef':m.coef_.flatten()
                    })
df.sort('coef',ascending=False,inplace=True)

print("Ridge Coefficients")
print("Most positive:")
print(df[0:50])
print("Most negative")
print(df[-50:])


# Lasso didn't work
# Scaling the data to unit variance didn't help

# Other things to try:
# Decision trees
#     http://scikit-learn.org/stable/modules/generated/sklearn.tree.DecisionTreeRegressor.html
# Random forests
#     http://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestRegressor.html
# Boosting
#     http://scikit-learn.org/stable/modules/generated/sklearn.ensemble.GradientBoostingRegressor.html
