#!/usr/bin/env python
"""
here we got the params to use in the pipelines
"""

def getting_params(data_original):
    """
    * data - data to clean and do the exploratory analysis
    * params - params to use in order to full fill the data

    Return the params of the data
    """
    data = data_original.copy()

    data['Embarked'].fillna('S', inplace=True)
    data['Cabin'].fillna(0, inplace=True)
    data['Age'].fillna(data['Age'].mean(), inplace=True)
    data['Fare'].fillna(data['Fare'].mean(), inplace=True)

    params = {}
    list_cols = data.columns.tolist()

    for col in list_cols:
        if col == 'Age':
            data[col] = data[col].astype(int)
        if col in ['Pclass', 'Sex', 'Embarked', 'Alone']:
            data[col] = data[col].astype('category')


    for col in list_cols:
        list_cat = data.select_dtypes(include=['category'])
        list_cat = list_cat.columns.tolist()
        continue_list = ['PassengerId', 'Name', 'Ticket', 'Cabin']

        if str(col) not in list_cat and str(col) not in continue_list:
            params[col] = data[col].mean()
            if col in ['SibSp', 'Parch']:
                params[col] = int(params[col])

        elif str(col) in continue_list:
             continue
        else:
            vals = data[col].value_counts()
            vals = vals.sort_values(ascending=False)
            vals = vals.reset_index()
            vals = vals.iloc[0, 0]
            params[col] = vals

    new = {}
    for col in params.keys():
        if col in ['Pclass', 'Sex', 'Embarked']:
            result = data[col].unique().to_list()
            result = [str(elem) for elem in result]
            name = '{}_values'.format(col)
            new[name] = result

    params.update(new)
    return params