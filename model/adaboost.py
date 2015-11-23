#
# MDST Ratings Analysis Challenge
# Model selection with Cross Validation
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

from sklearn import cross_validation
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import AdaBoostRegressor
from sklearn.preprocessing import Imputer
from sklearn.feature_extraction.text import CountVectorizer
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

# Select alpha with a validation set
Xtr, Xval, Ytr, Yval = cross_validation.train_test_split(
    Xtrain,
    Ytrain,
    test_size = 0.3,
    random_state = 0)

# Define window to search for alpha
# Cs = np.arange(10,15)  # Here C is max_depth
Cs = [8]

# Store MSEs here for plotting
mseTr = np.zeros((len(Cs),))
mseVal = np.zeros((len(Cs),))

# Search for lowest validation accuracy
print 'cross-validation'
for i in range(len(Cs)):
    print "C =", Cs[i]
    m = AdaBoostRegressor(DecisionTreeRegressor(max_depth=Cs[i]),
                            n_estimators=100)
    m.fit(Xtr, Ytr)
    YhatTr = m.predict(Xtr)
    YhatVal = m.predict(Xval)
    mseTr[i] = np.sqrt(mean_squared_error(YhatTr, Ytr))
    mseVal[i] = np.sqrt(mean_squared_error(YhatVal, Yval))
    print Cs[i], mseTr[i], mseVal[i]

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.plot(Cs, mseTr, hold=True)
plt.plot(Cs, mseVal)
plt.legend(['Training RMSE', 'Validation RMSE'])
plt.ylabel('RMSE')
plt.xlabel('max_depth')
plt.draw()
plt.savefig('adaboosting.png')

# Train new model using all of the training data
print 'Fit all training data'
m = AdaBoostRegressor(DecisionTreeRegressor(max_depth=8),
                        n_estimators=100)
m.fit(Xtrain, Ytrain)
Yhat = m.predict(Xtest)

# Save results in kaggle format
submit = pd.DataFrame(data={'id': TEST_ID, 'quality': Yhat})
submit.to_csv('adaboosting_submit.csv', index = False)



# Lasso didn't work
# Scaling the data to unit variance didn't help

# Other things to try:
# Random forests
#     http://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestRegressor.html
# Boosting
#     http://scikit-learn.org/stable/modules/generated/sklearn.ensemble.GradientBoostingRegressor.html
