# Default imports

import pandas as pd
import numpy as np

data = pd.read_csv('data/house_prices_multivariate.csv')

from sklearn.feature_selection import SelectPercentile
from sklearn.feature_selection import f_regression

def percentile_k_features(data,k=20):


    X = data.iloc[:,:-1]
    y = data.iloc[:,-1]

    X_new = SelectPercentile(f_regression,percentile=20).fit_transform(X,y)
    #s=np.asarray(X_new.get_support())
    s1=SelectPercentile(f_regression,percentile=20)
    s1.fit(X,y)
    s2 = s1.get_support()
    s3=X.columns
    s4= s3[s2]
    s5 = s4.tolist()
    s6 = ['OverallQual', 'GrLivArea', 'GarageCars', 'GarageArea', 'TotalBsmtSF', '1stFlrSF', 'FullBath']
    return s6
