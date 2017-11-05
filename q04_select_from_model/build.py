# Default imports
from sklearn.feature_selection import SelectFromModel
from sklearn.ensemble import RandomForestClassifier
import pandas as pd
import numpy as np

data = pd.read_csv('data/house_prices_multivariate.csv')

def select_from_model(data):
    X = data.iloc[:,:-1]
    y = data.iloc[:,-1]

    m = SelectFromModel(estimator=RandomForestClassifier())
    m.fit_transform(X,y)
    a = X.columns
    b = a[m.get_support()]
    c = b.tolist()
    return c
