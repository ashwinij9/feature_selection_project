# Default imports
import pandas as pd
import numpy as np

data = pd.read_csv('data/house_prices_multivariate.csv')

from sklearn.feature_selection import RFE
from sklearn.ensemble import RandomForestClassifier

def rf_rfe(data):
    X = data.iloc[:,:-1]
    y=data.iloc[:,-1]
    m = RFE(estimator=RandomForestClassifier())
    m.fit(X,y)
    a = X.columns
    b= a[m.support_]
    c= np.asarray(b)
    d= c.tolist()
    return d
