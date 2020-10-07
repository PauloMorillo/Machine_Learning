#!/usr/bin/env python
"""
Creating a model to run the Titanic Dataset
"""

# ************************************* import of packages ************************************
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import sys
import mlflow
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
import numpy as np
from sklearn import metrics
from urllib.parse import urlparse
from datetime import datetime
import os

# ******************************************* EDA ***********************************************
def Exploratory_analysis(data, debug):
    # filling Null Values
    if debug == 'Super':
        print(data.columns)
        print(data.isnull().sum())
    data['Embarked'].fillna('S', inplace=True)
    data['Cabin'].fillna(0, inplace=True)
    data['Age'].fillna(data['Age'].mean(), inplace=True)
    data['Fare'].fillna(data['Fare'].mean(), inplace=True)

    if debug == 'Super':
        print(data.isnull().sum())

        # Distribution of variables in data
        fig, ax = plt.subplots(4, 2, figsize=(20,10))
        plt.subplots_adjust(hspace=0.5)

        data['Survived'].value_counts().plot.bar(ax=ax[0,0])
        ax[0, 0].set_title('Survivors and died')

        data['Pclass'].value_counts().plot.bar(ax=ax[0, 1], color='purple')
        ax[0, 1].set_title('Number of persons per Pclass')

        data['Sex'].value_counts().plot.bar(ax=ax[1, 0], color='green')
        ax[1, 0].set_title('Distribution of Sex')
        ax[1, 0].tick_params(labelrotation=0)

        data['Age'].plot.hist(ax=ax[1, 1])
        ax[1, 1].set_title('Distribution Age')

        data['SibSp'].value_counts().plot.bar(ax=ax[2, 0], color='orange')
        ax[2, 0].set_title('number sibling or spouses')

        data['Parch'].value_counts().plot.bar(ax=ax[2, 1], color='blue')
        ax[2, 1].set_title('number of parents and children')

        data['Fare'].plot(ax=ax[3, 0], color='red')
        ax[3, 0].set_title('distribution Fare')

        data['Embarked'].value_counts().plot.bar(ax=ax[3, 1])
        ax[3, 1].set_title('number of passanger by Embarked')
        plt.show()

        # Some Correlations
        sns.countplot(data=data, x='Pclass', hue='Survived')
        plt.show()
        sns.countplot(data=data, x='Embarked', hue='Survived')
        plt.show()
        sns.scatterplot(data=data, x='Age', y='Fare', hue='Survived')
        plt.show()
        sns.boxplot(data=data, x='Sex', y='Fare', hue='Survived')
        plt.show()
        sns.countplot(data=data, x='SibSp', hue='Survived')
        plt.show()
        sns.countplot(data=data, x='Parch', hue='Survived')
        plt.show()

    data['Alone'] = data['Parch'] + data['SibSp']
    data['Alone'] = data['Alone'].apply(lambda x: 1 if x == 0 else 0)
    # removing some data that will no affect our training
    data.drop(['PassengerId', 'Name', 'Ticket', 'Cabin', 'Parch', 'SibSp'], axis=1, inplace=True)

    def change_gender(x):
        if x == 'male':
            return 0
        elif x == 'female':
            return 1

    data['Sex'] = data['Sex'].apply(change_gender)

    change = {'S': 1, 'C': 2, 'Q': 0}
    data['Embarked'] = data['Embarked'].map(change)

    return data

# *************************************** MAIN **********************************************
if __name__ == '__main__':
    num_commands = len(sys.argv)

    # default parameters
    path = '../data/titanic_data/'
    debug = 'Off'
    type = 'RandomForest'

    # conditions to fill the parameters
    if num_commands > 4:
        print("""Usage should be {debug_mode}, {path}, {type}
        ----> path: path to your titanic data
        ----> debug: debug mode, values On, Off, Super to see all graphs too
        ----> type: model to use in training possible values "RandomFores", "ElasticNet", "LogisticRegression"
        
        Note: if you want to use the default value use "_" in the parameter position""")
        exit(0)

    if num_commands > 1:
        debug = sys.argv[1]
        only = ['Off', 'On', 'Super', '_']
        if debug not in only:
            print('debug: debug mode, values On, Off, Super to see all graphs too')
            exit(0)
        if debug == '_':
            debug = 'Off'

    if num_commands > 2:
        path = sys.argv[2]
        if path == '_':
            path = '../data/titanic_data/'

    if num_commands > 3:
        list_models = ['RandomForest', 'KNeighbors', 'LogisticRegression', '_']
        type = sys.argv[3]
        if type not in list_models:
            print('type: model to use in training possible values "RandomForest", "KNeighbors", "LogisticRegression"')
            exit(0)
        if type == '_':
            type = 'RandomForest'


    train = pd.read_csv(path + 'train.csv')
    test = pd.read_csv(path + 'test.csv')
    aux_data = pd.read_csv(path + 'gender_submission.csv')
    test = pd.merge(test, aux_data, on='PassengerId', how='left')

    if debug == 'Super':
        print('first 5 rows of train data', train.head())

    # getting Train and Test cleaned
    train_clean = Exploratory_analysis(train, debug)
    test_clean = Exploratory_analysis(test, debug)

    if debug == 'Super':
        print('first 5 rows of train clean data', train_clean.head())
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

        if type == 'RandomForest':
            print('RANDOM FOREST MODEL')
            n_estimators = 250
            max_depth = 9
            if debug in ['On', 'Super']:
                n_estimators = input('insert a number for  n_estimators parameter  or "_" to default: ')
                max_depth = input('insert a number between 0-9 for max_depth  parameter or "_" to default: ')
                if n_estimators == '_':
                    n_estimators = 250
                else:
                    n_estimators = int(n_estimators)
                if max_depth == '_':
                    max_depth = 9
                else:
                    max_depth = int(max_depth)

            model = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth)

        if type == 'LogisticRegression':
            print('LOGISTIC REGRESSION MODEL')
            max_iter = 10000

            if debug in ['On', 'Super']:
                max_iter = input('insert a number for max_iter parameter  or "_" to default: ')
                if max_iter == '_':
                    max_iter = 10000
                else:
                    max_iter = int(max_iter)

            model = LogisticRegression(max_iter = max_iter)


        if type == 'KNeighbors':
            print('K NEIGHBORS MODEL')
            n_neighbors = 5

            if debug in ['On', 'Super']:
                n_neighbors = input('insert a number for n_neighbors  parameter  or "_" to default: ')
                if n_neighbors  == '_':
                    n_neighbors  = 5
                else:
                    n_neighbors  = int(n_neighbors )

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