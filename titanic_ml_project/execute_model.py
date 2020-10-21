#!/usr/bin/env python
"""
Creating a model to run the Titanic Dataset
"""

# ************************************* import of packages ************************************
import pandas as pd
import mlflow
from sklearn.metrics import f1_score, accuracy_score
from urllib.parse import urlparse
import os
import numpy as np
from preprocessing_data import getting_params
from pipeline_preprocessing import pipeline
from getting_dict import get_dict
from datetime import datetime


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

    with mlflow.start_run():
        model = pipeline(params, model_type, models_dict)
        model.fit(X_train, Y_train)
        pred_train = model.predict(X_train)
        pred_test = model.predict(X_test)
        f1_train = f1_score(Y_train, pred_train)
        accuracy_train = accuracy_score(Y_train, pred_train)
        f1_test = f1_score(Y_test, pred_test)
        accuracy_test = accuracy_score(Y_test, pred_test)
        print("Our train results:", f1_train, accuracy_train)
        print("Our test results:", f1_test, accuracy_test)
        for key_m, values in models_dict.items():
            if key_m == model_type:
                for key, value in values.items():
                    mlflow.log_param(key, value)
            else:
                for key, value in values.items():
                    mlflow.log_param(key, "None")

        mlflow.log_metric("f1_train", f1_train)
        mlflow.log_metric("f1_test", f1_test)
        mlflow.log_metric("Accuracy_train", accuracy_train)
        mlflow.log_metric("Accuracy_test", accuracy_test)
        tracking_url_type_store = urlparse(mlflow.get_tracking_uri()).scheme
        # Model registry does not work with file store
        if tracking_url_type_store != "file":

            # Register the model
            # There are other ways to use the Model Registry, which depends on the use case,
            # please refer to the doc for more information:
            # https://mlflow.org/docs/latest/model-registry.html#api-workflow
            mlflow.sklearn.log_model(model, "model", registered_model_name=model_type)
        else:
            mlflow.sklearn.log_model(model, "model")

        now = datetime.now()
        dt_string = now.strftime("%d_%m_%Y_%H_%M_%S")
        name = model_type + '_' + dt_string
        if not os.path.exists('./models'):
            os.mkdir('models')
        mlflow.sklearn.save_model(model, "./models/{}".format(name))
