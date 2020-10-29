# Titanic Dataset, Training, Models and Mlflow

![](https://www.ecestaticos.com/image/clipping/44ad0adfdaca39bd1c9d5de802e1bb07/que-paso-de-verdad-con-los-cadaveres-de-los-pasajeros-pobres-del-titanic.jpg)

## Authors

- Paulo Morillo
- Jorge Enrique Zafra

## Requirements

python version:

- Python 3.8

You will have to install the next packages in order to run the code

- mlflow
- scikit learn
- pandas
- matplotlib
- seaborn
- numpy
- docker
- FastApi

or just install the project dependencies with
`pip install -r requirements.txt`

## Usage

This is how you can run the code:

```
python main.py
```

With this line of commands you will be able to train the Titanic DataSet with the default parameters and they are:

- model: Random Forest
- - n_estimators: 250
- - max_depth: 9

if you want to add more parameters you will have to add params in the execute_model.py file where you will find variables like:

- n_estimators = 'None'
- max_depth = 'None'
- max_iter = 'None'

right in that part you can add your parameters with none because you need to declare all in None because of the MLflow interface. Also remember to add your new parameters inside the setup_models.yaml

## Our Models

Support 3 different models:

- Random Forest
- Kneighbors
- logistic Regression

so if you want to use another model instead of the default one (Random forest) you can do<br>
`python main.py --type KNeighbors`<br>
`python main.py --type LogisticRegression`

## MLFLOW

Once you install all the requirements and run one or more models, you will be able to run mlflow to see your models and compare them.

`mlflow ui`

## DOCKER

we create a DockerFile which will have all the necesary to run our program into a server.

create the image running docker compose like
`docker compose up`<br>

doing this it will get the dockerfile and it will built it.

<b>NOTE</b> everytime you change your program and want to upload the image remember to create again the image with the latest changes that you made. in other way it will upload with the original customization

## API

to run the API you can do it locally with:

- uvicorn main:app --reload

unless you upload the image of docker into a server in that way you will run the uvicorn using the 8000 port

## Other

Also if you have some problems you can check step_by_step file which is a doc where we put some problems that appeared while we created the project and how we solved them. maybe it can help you or maybe not
