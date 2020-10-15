#!/usr/bin/env python
"""
Creating a model to run the Titanic Dataset
"""

# ************************************* import of packages ************************************
import pandas as pd
import mlflow
from sklearn.metrics import f1_score, accuracy_score

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
# from .pipeline_preprocessing import pipeline
from getting_dict import get_dict


def getting_data(path):
    """
    This method read a csv file from variable path
    """
    train = pd.read_csv(path + 'train.csv')
    test = pd.read_csv(path + 'test.csv')
    aux_data = pd.read_csv(path + 'gender_submission.csv')
    test = pd.merge(test, aux_data, on='PassengerId', how='left')

    # getting Train and Validation Sets
    Y_train = train['Survived']
    X_train = train.drop(['Survived'], axis=1)

    Y_test = test['Survived']
    X_test = test.drop(['Survived'], axis=1)
    return (X_train, Y_train, X_test, Y_test)


# *************************************** MAIN **********************************************
def execute_m(path, debug, model_type):
    """
    This method execute the model
    """
    # pd.set_option('display.max_rows', 500)
    # pd.set_option('display.max_columns', 500)
    # default parameters:
    name = type
    n_estimators = 'None'
    max_depth = 'None'
    max_iter = 'None'
    n_neighbors = 'None'
    models_dict = get_dict("setup_models.yaml")
    X_train, Y_train, X_test, Y_test = getting_data(path)
    params = getting_params(X_train)
    model = pipeline(params, model_type, models_dict)
    model.fit(X_train, Y_train)
    pred_train = model.predict(X_train)
    pred_test = model.predict(X_test)
    f1_train = f1_score(Y_train, pred_train, average=None)
    accuracy_train = accuracy_score(Y_train, pred_train)
    f1_test = f1_score(Y_test, pred_test, average=None)
    accuracy_test = accuracy_score(Y_test, pred_test)
    print("Our train results:", f1_train, accuracy_train)
    print("Our test results:", f1_test, accuracy_test)
