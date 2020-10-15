#!/usr/bin/env python
"""
This module the pipeline preprocessing
in this way we are going to create a model with
custom preprocessing
"""

from sklearn.pipeline import Pipeline
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier


# ******************************************* Preprocessing ***********************************************
def pipeline(params, model_type, models_dict):
    """
    here we are going to create the different pipelines per Model
        """

    # dropping not desired features
    class clean_trans(BaseEstimator, TransformerMixin):
        def __init__(self):
            pass

        def fit(self, X, y=None):
            return self

        def transform(self, X, y=None):
            X_ = X.copy()
            X_.drop(['PassengerId', 'Name', 'Ticket', 'Cabin'], axis=1, inplace=True)
            return X_

    # preprocessing
    class prep(BaseEstimator, TransformerMixin):
        def __init__(self, par):
            self.par = par

        def fit(self, X, y=None):
            return self

        def transform(self, X, y=None):
            X_ = X.copy()
            list_cols = X_.columns.tolist()

            for col in list_cols:
                if col in ['Pclass', 'Sex', 'Embarked', 'Alone']:
                    X_[col] = X_[col].astype('category')

            # fill Null
            params = self.par
            for elem in params.keys():
                if not 'value' in elem:
                    X_[elem].fillna(params[elem], inplace=True)

            for elem in params.keys():
                if elem == 'Age':
                    X_[elem] = X_[elem].apply(lambda x: x if x > 0 or x < 120 else params[elem])

                elif elem == 'Fare':
                    X_[elem] = X_[elem].apply(lambda x: x if x >= 0 else params[elem])

                elif elem in ['SibSp', 'Parch']:
                    def change(x):
                        if type(x) is int:
                            if x >= 0:
                                return x
                            else:
                                return params[elem]
                        else:
                            if x >= 0:
                                return int(x)
                            else:
                                return params[elem]

                    X_[elem] = X_[elem].apply(change)

                elif elem in ['Pclass', 'Sex', 'Embarked']:
                    key_value = '{}_values'.format(elem)
                    X_[elem] = X_[elem].apply(lambda x: str(x) if str(x) in params[key_value] else params[elem])

            return X_

    class random_prep(BaseEstimator, TransformerMixin):
        def __init__(self):
            pass

        def fit(self, X, y=None):
            return self

        def transform(self, X, y=None):
            X_ = X.copy()
            X_['Sex'] = X_['Sex'].map({'male': '1', 'female': '0'})
            X_['Embarked'] = X_['Embarked'].map({'S': '1', 'C': '2', 'Q': '3'})
            return X_

    if model_type in ['RandomForest']:
        pipe_random = Pipeline(steps=[
            ('clean', clean_trans()),
            ('prep', prep(params)),
            ('random_prep', random_prep()),
            ('model', RandomForestClassifier(
                max_depth=models_dict[model_type]["max_depth"],
                n_estimators=models_dict[model_type]["n_estimators"]
            ))
        ])
    if model_type in ['LogisticRegression']:
        pipe_random = Pipeline(steps=[
            ('clean', clean_trans()),
            ('prep', prep(params)),
            ('random_prep', random_prep()),
            ('model', LogisticRegression(
                max_iter=models_dict[model_type]["max_iter"]
            ))
        ])
    if model_type in ['KNeighbors']:
        pipe_random = Pipeline(steps=[
            ('clean', clean_trans()),
            ('prep', prep(params)),
            ('random_prep', random_prep()),
            ('model', KNeighborsClassifier(
                n_neighbors=models_dict[model_type]["n_neighbors"]
            ))
        ])
    return pipe_random
