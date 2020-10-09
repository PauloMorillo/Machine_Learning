#!/usr/bin/env python
"""
This module has the Exploratory_analysis(data, debug) method
"""

# ************************************* import of packages ************************************
import matplotlib.pyplot as plt
import seaborn as sns


# ******************************************* EDA ***********************************************
def Exploratory_analysis(data, debug):
    """
    * data - data to clean and do the exploratory analysis
    * debug - to know if we have to show all the EDA

    Return the cleaned data
    """
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
        fig, ax = plt.subplots(4, 2, figsize=(20, 10))
        plt.subplots_adjust(hspace=0.5)

        data['Survived'].value_counts().plot.bar(ax=ax[0, 0])
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
