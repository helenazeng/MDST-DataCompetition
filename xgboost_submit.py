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
import logging
from time import localtime, strftime

import pandas as pd
import numpy as np
from scipy.sparse import csr_matrix, hstack

from sklearn import cross_validation
from sklearn.preprocessing import Imputer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import mean_squared_error

import xgboost as xgb

np.random.seed(0)

logging.basicConfig(filename='xgboost ' + strftime("%Y-%m-%d %H:%M:%S", localtime()) + '.log',
                    level=logging.INFO,
                    format='%(asctime)s %(message)s')

# Load in the data - pandas DataFrame objects
print 'Load data'
rats_tr = pd.read_csv('data/newtrain.csv')
rats_te = pd.read_csv('data/newtest.csv')

# Construct bigram representation
print 'Construct bigram representation'
count_vect = CountVectorizer(min_df=10,ngram_range=(1,2))

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

Xtrain = xgb.DMatrix(Xtrain, label=Ytrain)
Xtest = xgb.DMatrix(Xtest)

# Train new model using all of the training data
print 'Fit all training data'
param = {'max_depth': 6,
        'eta': 0.2,
        'silent': 0,
        'objective': 'reg:linear',
        'eval_metric': 'rmse' }
num_round = 750
logging.info("C = %d", 6)
m = xgb.train(param, Xtrain, num_round)

Yhat = m.predict(Xtest)

# Save results in kaggle format
submit = pd.DataFrame(data={'id': TEST_ID, 'quality': Yhat})
submit.to_csv('xgboost_submit.csv', index = False)



# Lasso didn't work
# Scaling the data to unit variance didn't help

# Other things to try:
# Random forests
#     http://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestRegressor.html
# Boosting
#     http://scikit-learn.org/stable/modules/generated/sklearn.ensemble.GradientBoostingRegressor.html
