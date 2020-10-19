#!/usr/bin/env python
"""
All begins here
"""

# ************************************* import of packages ************************************
import typer
from execute_model import execute_m
import mlflow.sklearn
from mlflow.entities import ViewType
from fastapi import FastAPI
from typing import List, Dict
import pandas as pd


# from mlflow.tracking.client import MlflowClient


app = FastAPI()


@app.post("/predict/", response_model=List[Dict], response_model_exclude_unset=True)
async def create_item(data: list):
    """
    This method gets column and data
    column: features
    data: values of features
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

    best_model = mlflow. search_runs(experiment_ids="0", run_view_type=ViewType.ACTIVE_ONLY,
                                    max_results=1, order_by=["metrics.test_F1 DESC"])

    model_id = best_model.loc[:, "run_id"].values[0]
    model = mlflow.sklearn.load_model("/usr/src/app/mlruns/0/" + model_id + "/artifacts/model")
    df = data
    my_dict = [elem.copy() for elem in data]

    df = pd.DataFrame(df)
    prediction = model.predict(df)
    prediction = list(prediction)
    print(prediction)
    for i in range(len(my_dict)):
        my_dict[i]['Prediction'] = str(prediction[i])
    return my_dict


# *************************************** MAIN **********************************************
def main(path: str = '../data/titanic_data/',
         model: str = typer.Option(
             "RandomForest", help="RandomForest, KNeighbors or LogisticRegression"),
         debug: bool = False):
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

    list_models = ['RandomForest', 'KNeighbors', 'LogisticRegression']
    if model not in list_models:
        print("type: model to use in training possible values ", end=" ")
        # in theory, It's printing with style, but I couldn't try
        valid = typer.secho("RandomForest, KNeighbors or LogisticRegression",
                            fg=typer.colors.GREEN, bold=True)
        exit(0)
    execute_m(path, debug, model)


if __name__ == '__main__':
    typer.run(main)
