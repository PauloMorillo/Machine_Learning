#!/usr/bin/env python
"""
All begins here
"""

# ************************************* import of packages ************************************
import typer
from execute_model import execute_m


# *************************************** MAIN **********************************************
def main(path: str = '../data/titanic_data/',
         type: str = typer.Option(
             "RandomForest", help="RandomForest, KNeighbors or LogisticRegression"),
         debug: bool = False, Super: bool = False):
    """ This is our main method """
    # print(path, type, debug, Super) # this print show the arguments
    # conditions to fill the parameters
    list_models = ['RandomForest', 'KNeighbors', 'LogisticRegression', '_']
    if type not in list_models:
        print("type: model to use in training possible values ", end=" ")
        # in theory, It's printing with style, but I couldn't try
        valid = typer.secho("RandomForest, KNeighbors or LogisticRegression",
                            fg=typer.colors.GREEN, bold=True)
        exit(0)
    execute_m(path, debug, type, Super=False)

if __name__ == '__main__':
    typer.run(main)
