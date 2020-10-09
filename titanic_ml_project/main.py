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
         debug: bool = False,
         super: bool = False):

    """
    This is our method to use typer
    * default behaviour
      - ./main.py
    * to use flags
      - ./main.py --path mypath --type model_type
    * to use bool flags we have to call them or we are goin to have them with False
      - ./main.py --path mypath --type model_type --debug
      - in this way debug is going ot be True without the flag False
    """
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
