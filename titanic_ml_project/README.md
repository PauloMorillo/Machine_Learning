# Titanic Dataset, Training, Models and Mlflow
![](https://www.ecestaticos.com/image/clipping/44ad0adfdaca39bd1c9d5de802e1bb07/que-paso-de-verdad-con-los-cadaveres-de-los-pasajeros-pobres-del-titanic.jpg)
## Authors
Jorge Enrique Zafra - [LinkedIn](https://www.linkedin.com/in/jorge-enrique-zafra-ria%C3%B1o-49268193/)
## Requirements

my python version is
- Python 3.8

You will have to install the next packages in order to run the code
- mlflow
- scikit learn
- pandas
- matplotlib
- seaborn
- numpy

or just install the project dependencies with
`pip install -r requirements.txt`

## Usage
### Basic Usage
This is how you can run the code:
```
python execute_model.py
```
With this line of commands you will be able to train the Titanic DataSet with the default parameters and they are:
- debug_mode = 'Off'
- path = '../data/titanic_data'
- model: 'Random Forest' with 'n_estimators=250' and 'max_depth=9'

### Advanced Usage
This is how the Advanced Mode works
```
python execute_model {debug_mode} {path} {type}
```
with <b>debug_mode</b> you will have the next options:
- Off: just to see the output with the default parameters
- On: this option let you play with the 


