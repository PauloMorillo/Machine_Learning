#!/usr/bin/env python
"""
This module the pipeline preprocessing
in this way we are going to create a model with
custom preprocessing
"""

from sklearn.pipeline import Pipeline
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.ensemble import RandomForestClassifier

# ******************************************* Preprocessing ***********************************************
def pipeline(params):
    """
    here we are going to create the different pipelines per Model
        """

    # dropping not desired features
    class clean_trans(BaseEstimator, TransformerMixin):
        def __init__(self):
            pass

        def fit(self, X, y = None):
            return self

        def transform(self, X, y = None):
            X_ = X.copy()
            X_.drop(['PassengerId', 'Name', 'Ticket', 'Cabin'], axis=1, inplace=True)
            return X_

    # preprocessing
    class prep(BaseEstimator, TransformerMixin):
        def __init__(self, par):
            self.par = par

        def fit(self, X, y = None):
            return self

        def transform(self, X, y = None):
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

    pipe_random = Pipeline(steps=[
        ('clean', clean_trans()),
        ('prep', prep(params)),
        ('model', RandomForestClassifier())
    ])

    return pipe_random