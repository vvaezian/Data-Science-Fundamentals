
#### confusion_matrix 
to evaluate the quality of the output of a classifier

#### Misc.
````Python
scaler = MinMaxScaler(feature_range=(0,1))
scaled_train_samples = scaler.fit_transform((train_samples).reshape(-1, 1)) # reshape is to turn it into suitable format for keras
````
- After fitting a model we can use `.coef_` and `.intercept_` to get the coefficients and intercept of the fitted curve.

#### ParameterGrid (get results for combinations of given parameters)
````Python
from sklearn.model_selection import ParameterGrid
params_dict = {
    #, 'learning_rate':np.arange(.01, 1, .01) ###
    #, 'max_depth':[1,2,3, 4, 5]   
    #, 'n_estimators':range(100, 300, 20)
    #, 'gamma':range(0, 1500, 100)
    #, 'min_child_weight':np.arange(0, 2, .1)
    #, 'subsample':np.arange(0, 1.1, .1)
     'reg_alpha' : np.arange(0, 5000, 100)
     ,'reg_lambda':np.arange(0, 1, .01)
     ,'base_score' : range(0, 10000, 500) 
    #, 'objective':['reg:linear', 'reg:gamma', 'reg:tweedie']
    #, 'booster':['gbtree', 'gblinear', 'dart']
    #, 'scale_pos_weight' : np.arange(.9, 1.1, .02) # default is best
    #, 'max_delta_step':[0, 10, 100]  # constant
            }

import math
best_MAE = math.inf # Python 3.5+
best_params = {}

import time
from tqdm import tqdm_notebook as tqdm  # for progressbar

for params in tqdm(ParameterGrid(params_dict)):
    XGB_model = XGBRegressor(**params,  random_state=0)
    #print(XGB_model)
    XGB_model.fit(X_train, y_train)
    MAE = round(mean_absolute_error(XGB_model.predict(X_val), y_val), 2)
    if MAE < best_MAE:
        print(params, '\n', MAE)
        best_params = params
        best_MAE = MAE
````
#### Get ressults for testing one parameter at a time
````Python
params_dict = {   'n_estimators': range(20, 300, 5)
                , 'learning_rate': np.arange(.01, 1, .01)
                , 'gamma': range(0, 20000, 100)
                , 'base_score': range(0, 10000, 100)
                , 'reg_alpha': np.arange(0, 3000, 10)
                , 'reg_lambda': np.arange(0, 3, .01)
                , 'max_delta_step':[0, 10, 100]
                , 'objective':['reg:linear', 'reg:gamma', 'reg:tweedie']
                , 'booster':['gbtree', 'gblinear', 'dart']
                , 'min_child_weight':np.arange(0, 2, .1)
                , 'scale_pos_weight' : np.arange(.9, 1.1, .02)
                , 'subsample':np.arange(0, 1.1, .1)
              }

for param_name, param_range in params_dict.items():
    y_list = []
    for p in param_range:
        param = {param_name:p}
        XGB_model2 = XGBRegressor(**param, max_depth=2)
        y_list.append(runManyTimes(XGB_model2))
    plt.figure(figsize=(10,5))
    plt.xlabel(param_name)
    plt.ylabel('MAE')
    plt.plot(param_range, y_list)
    plt.show()
````
#### Label Encoding
````Python
from sklearn import preprocessing
le = preprocessing.LabelEncoder()
Store_Desc_numeric = le.fit_transform(data.Store_Desc)
data['Store_Desc'] = Store_Desc_numeric
````
