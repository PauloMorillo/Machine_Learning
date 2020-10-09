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
from preprocessing_data import Exploratory_analysis
from getting_dict import get_dict


# *************************************** MAIN **********************************************
def execute_m(path, debug, type, Super=False):
    train = pd.read_csv(path + 'train.csv')
    test = pd.read_csv(path + 'test.csv')
    aux_data = pd.read_csv(path + 'gender_submission.csv')
    test = pd.merge(test, aux_data, on='PassengerId', how='left')

    if Super:
        print('first 5 rows of train data', train.head())

    # getting Train and Test cleaned
    train_clean = Exploratory_analysis(train, debug)
    test_clean = Exploratory_analysis(test, debug)

    if Super:
        print('first 5 rows of train clean data\n', train_clean.head())
        print('first 5 rows of test clean data', test_clean.head())

    # getting Train and Validation Sets
    Y_train = train['Survived']
    X_train = train.drop(['Survived'], axis=1)

    Y_test = test_clean['Survived']
    X_test = test_clean.drop(['Survived'], axis=1)

    with mlflow.start_run():
        # default parameters:
        name = type
        n_estimators = 'None'
        max_depth = 'None'
        max_iter = 'None'
        n_neighbors = 'None'
        models_dict = get_dict("setup_models.yaml")
        print(models_dict)  # to see the dictionary

        if type == 'RandomForest':
            print('RANDOM FOREST MODEL')
            n_estimators = 250
            max_depth = 9
            if debug or Super:
                n_estimators = models_dict[type]["n_estimators"]
                max_depth = models_dict[type]["max_depth"]
                n_estimators = int(n_estimators)
                max_depth = int(max_depth)

            model = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth)

        if type == 'LogisticRegression':
            print('LOGISTIC REGRESSION MODEL')
            max_iter = 10000

            if debug or Super:
                max_iter = models_dict[type]["max_iter"]
                max_iter = int(max_iter)

            model = LogisticRegression(max_iter=max_iter)

        if type == 'KNeighbors':
            print('K NEIGHBORS MODEL')
            n_neighbors = 5

            if debug or Super:
                n_neighbors = models_dict[type]["n_neighbors"]
                n_neighbors = int(n_neighbors)

            model = KNeighborsClassifier(n_neighbors=n_neighbors)

        model.fit(X_train, Y_train)
        print('TRAIN')
        train_acc = model.score(X_train, Y_train)
        print('ACC: ', train_acc)

        Y_pred = model.predict(X_train)
        F1_score_train = np.round(metrics.f1_score(Y_train, Y_pred) * 100, 2)
        print('F1 score : ', F1_score_train)

        print('TEST')
        test_acc = model.score(X_test, Y_test)
        print('ACC: ', test_acc)
        Y_pred = model.predict(X_test)
        F1_score_test = np.round(metrics.f1_score(Y_test, Y_pred) * 100, 2)
        print('F1 score : ', F1_score_test)

        mlflow.log_param("n_estimators", n_estimators)
        mlflow.log_param("max_depth", max_depth)
        mlflow.log_param("max_iter", max_iter)
        mlflow.log_param("n_neighbors", n_neighbors)

        mlflow.log_metric(" tarin_acc", train_acc)
        mlflow.log_metric("train_F1", F1_score_train)

        mlflow.log_metric("test_acc", test_acc)
        mlflow.log_metric("test_F1", F1_score_test)

        tracking_url_type_store = urlparse(mlflow.get_tracking_uri()).scheme
        # Model registry does not work with file store
        if tracking_url_type_store != "file":

            # Register the model
            # There are other ways to use the Model Registry, which depends on the use case,
            # please refer to the doc for more information:
            # https://mlflow.org/docs/latest/model-registry.html#api-workflow
            mlflow.sklearn.log_model(model, "model", registered_model_name=name)
        else:
            mlflow.sklearn.log_model(model, "model")

        now = datetime.now()
        dt_string = now.strftime("%d_%m_%Y_%H_%M_%S")
        name = name + '_' + dt_string
        if not os.path.exists('./models'):
            os.mkdir('models')
        mlflow.sklearn.save_model(model, "./models/{}".format(name))
