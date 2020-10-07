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
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import numpy as np
from sklearn import metrics

# ******************************************* EDA ***********************************************
def Exploratory_analysis(data, debug):
    # filling Null Values
    if debug == 'Super':
        print(data.columns)
        print(data.isnull().sum())
    data['Embarked'].fillna(method='ffill', inplace=True)
    data['Cabin'].fillna(0, inplace=True)
    data['Age'].fillna(data['Age'].median(), inplace=True)
    data['Fare'].fillna(data['Fare'].median(), inplace=True)

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

    # removing some data that will no affect our training
    data.drop(['PassengerId', 'Name', 'Ticket', 'Cabin'], axis=1, inplace=True)

    def change_gender(x):
        if x == 'male':
            return 0
        elif x == 'female':
            return 1

    data['Sex'] = data['Sex'].apply(change_gender)

    change = {'S': 1, 'C': 2, 'Q': 0}
    data['Embarked'] = data['Embarked'].map(change)

    return data

# ******************************** Model performance ******************************
def eval_metrics(actual, pred):
    rmse = np.sqrt(mean_squared_error(actual, pred))
    mae = mean_absolute_error(actual, pred)
    r2 = r2_score(actual, pred)
    return rmse, mae, r2


if __name__ == '__main__':
    num_commands = len(sys.argv)

    # default parameters
    path = '../data/titanic_data/'
    debug = 'Off'
    type = 'RandomForest'

    # conditions to fill the parameters
    if num_commands > 4:
        print('Usage should be {debug}, {path}, {type}')
        print('----> path: path to your titanic data')
        print('----> debug: debug mode, values On, Off, Super to see all graphs too')
        print('----> type: model to use in training possible values "RandomFores", "LinearRegression", "DNN"')
        exit(0)

    if num_commands > 1:
        debug = sys.argv[1]
        only = ['Off', 'On', 'Super', '_']
        if debug not in only:
            print('debug: debug mode, values On, Off, Super to see all graphs too')
            exit(0)
        debug = 'Off'

    if num_commands > 2:
        path = sys.argv[2]
        if path == '_':
            path = '../data/titanic_data/'

    if num_commands > 3:
        list_models = ['RandomForest', 'LinearRegression', 'DNN', '_']
        type = sys.argv[3]
        if type not in list_models:
            print('type: model to use in training possible values "RandomFores", "LinearRegression", "DNN"')
            exit(0)
        if type == '_':
            type = 'RandomForest'


    train = pd.read_csv(path + 'train.csv')
    test = pd.read_csv(path + 'test.csv')

    if debug == 'Super':
        print(train.head())

    # getting Train and Test cleaned
    train_clean = Exploratory_analysis(train, debug)
    test_clean = Exploratory_analysis(test, 'Off')

    Y_train = train_clean['Survived']
    X_train = train_clean.drop(['Survived'], axis=1)

    X_test = test_clean

    with mlflow.start_run():
        if type == 'RandomForest':
            # train model
            random_forest = RandomForestClassifier(n_estimators=100)
            random_forest.fit(X_train, Y_train)
            Y_pred = random_forest.predict(X_train)
        if type == 'LinearRegression':
            print('elastic')
        if type == 'DNN':
            print('DeepNeNet')

        if debug == 'On' or debug == 'Super':
            # metrics
            print('MODEL USED {}'.format(type.upper()))
            print('Precision : ', np.round(metrics.precision_score(Y_train, Y_pred) * 100, 2))
            print('Accuracy : ', np.round(metrics.accuracy_score(Y_train, Y_pred) * 100, 2))
            print('Recall : ', np.round(metrics.recall_score(Y_train, Y_pred) * 100, 2))
            print('F1 score : ', np.round(metrics.f1_score(Y_train, Y_pred) * 100, 2))
            print('AUC : ', np.round(metrics.roc_auc_score(Y_train, Y_pred) * 100, 2))

            (rmse, mae, r2) = eval_metrics(Y_train, Y_pred )
            print("Randomforest model n_estimators{} ".format(100))
            print("  RMSE: %s" % rmse)
            print("  MAE: %s" % mae)
            print("  R2: %s" % r2)

        if debug == 'Off':
            print('MODEL USED {}'.format(type.upper()))
            print('F1 score : ', np.round(metrics.f1_score(Y_train, Y_pred) * 100, 2))
        # performance
        print('PREDICTION TEST')
        Y_pred_test = random_forest.predict(X_test)
        print(Y_pred_test)

