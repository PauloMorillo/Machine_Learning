#!/usr/bin/env python
"""
Creating a model to run the Titanic Dataset
"""

# ************************************* import of packages ************************************
import pandas as pd
import mlflow

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
import numpy as np
from sklearn import metrics
from urllib.parse import urlparse
from datetime import datetime
import os
from preprocessing_data import getting_params
from pipeline_preprocessing import pipeline
#from .pipeline_preprocessing import pipeline
from getting_dict import get_dict


# *************************************** MAIN **********************************************
def execute_m(path, debug, model_type, Super=False):
    # default parameters:
    name = type
    n_estimators = 'None'
    max_depth = 'None'
    max_iter = 'None'
    n_neighbors = 'None'
    models_dict = get_dict("setup_models.yaml")

    train = pd.read_csv(path + 'train.csv')
    test = pd.read_csv(path + 'test.csv')
    aux_data = pd.read_csv(path + 'gender_submission.csv')
    test = pd.merge(test, aux_data, on='PassengerId', how='left')
    pd.set_option('display.max_rows', 500)
    pd.set_option('display.max_columns', 500)

    # getting Train and Validation Sets
    Y_train = train['Survived']
    X_train = train.drop(['Survived'], axis=1)

    Y_test = test['Survived']
    X_test = test.drop(['Survived'], axis=1)

    params = getting_params(X_train)
    random = pipeline(params)

    random.fit(X_train, Y_train)
    pred = random.predict(X_test)
    print(pred)



