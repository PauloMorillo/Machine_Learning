#!/usr/bin/env python
"""
All begins here
"""

# ************************************* import of packages ************************************
import typer
from execute_model import execute_m
import os
import mlflow.sklearn
from mlflow.entities import ViewType
# from mlflow.tracking.client import MlflowClient


# *************************************** MAIN **********************************************
def main(path: str = '../data/titanic_data/',
         model_type: str = typer.Option(
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
    # ***************************** MlflowClient ***********************************+
    # if os.environ["ML_DEPLOY"]:
    #     run = MlflowClient().search_runs(
    #         experiment_ids="0",
    #         run_view_type=ViewType.ACTIVE_ONLY,
    #         max_results=1,
    #         order_by=["metrics.test_F1 DESC"])[0]
    #     run_id = run.__dict__.get('_info').__dict__.get('_run_id')
    #     model = mlflow.sklearn.load_model("runs:/" + run_id + "/model")
    #     print(model.predict([[1, 1, 27.0, 7.0000, 1, 1]]))

    # **************************** Dataframe mlflow ************************************
    if os.environ["ML_DEPLOY"]:
        best_model = mlflow.search_runs(experiment_ids="0", run_view_type=ViewType.ACTIVE_ONLY,
                                        max_results=1, order_by=["metrics.test_F1 DESC"])

        model_id = best_model.loc[:, "run_id"].values[0]
        model = mlflow.sklearn.load_model("runs:/" + model_id + "/model")
        print(model.predict([[1, 1, 27.0, 7.0000, 1, 1]]))

    else:
        list_models = ['RandomForest', 'KNeighbors', 'LogisticRegression', '_']
        if model_type not in list_models:
            print("type: model to use in training possible values ", end=" ")
            # in theory, It's printing with style, but I couldn't try
            valid = typer.secho("RandomForest, KNeighbors or LogisticRegression",
                                fg=typer.colors.GREEN, bold=True)
            exit(0)
        execute_m(path, debug, model_type, Super=False)


if __name__ == '__main__':
    typer.run(main)
