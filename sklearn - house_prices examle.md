````Python
import pandas as pd

data = pd.read_csv('../input/house-prices-advanced-regression-techniques/train.csv')
y = data.SalePrice
````
# Exploratory data analysis (EDA)
````Python
data.describe()  # shows min, max, mean, ... . This is applied only on numerical columns. For categorical ones see below.
data.colName.describe()
data[data.columns[6:10]].describe()
data.head() # top 5 rows (similarly data.tail())
categoricals = data.select_dtypes(exclude=[np.number])
categoricals.describe() # shows count (number of non-null values), unique, top (most commonly occurring value), freq (frequency of the top value)
data.info() # prints columns together with how many non-null value each has, and the type of its data.
            # by default the output is sent to stdout. To redirect it a buffer need to be defined.
data.shape # number of rows and columns
data.colName.unique()  # list of unique values of a (categorical) column
cols_with_missing_values = [col for col in data.columns if data[col].isnull().any()] # type: List
cols_with_missing_values = data.columns[data.isnull().any()] # type: Index
data.isnull().sum().sort_values(ascending=False)[:25] # columns with count of their missing values
data[null_cols].isnull().sum().sort_values(ascending=False)[:25] # columns that have missing value with count of their missing values
categoricals.describe().iloc[1].sort_values(ascending=False) # categorical columns with count of their unique values

print ("Skew is:", y.skew()) # skewness of target
plt.hist(y, bins=25)
plt.show()

# count of each unique value of a column (suitable for categorical data)
data.colName.value_counts()
````
# Exploring the Relationship between Columns 
**Correlation** between columns (shows only numeric columns)
````Python
corr = data.corr() 
print (corr['SalePrice'].sort_values(ascending=False)[:5], '\n')
print (corr['SalePrice'].sort_values(ascending=False)[-5:], '\n')
low_corr_cols = [ i for i in corr['SalePrice'].index if abs(corr['SalePrice'][i]) < 0.3]
# corr['SalePrice'] produces a Panda Series. s.index returns the indexes of the series s.

sns.heatmap(corr, annot=True)  # visualize the correlations using seaborn

corr = df.corr()['SalePrice']
corr[abs(corr)> .5]
````
Pairwise relationship of columns:
```
sns.jointplot(x='col1', y='col2', data=df)
```

average of SalePrice by Neighborhood  
````Python
Neighborhood_meanSalePrice = data.groupby('Neighborhood')['SalePrice'].mean().sort_values()
````
### Visualization ###
- **scatterplot** is good for numerical data  
````Python
plt.scatter(data.GrLivArea, np.log(data.SalePrice))
````
- **countplot** is counterpart of histogram for categorical data
````Python
import seaborn as sns
sns.countplot(data.SalePrice)   
````
- **stripplot**  
Camparing a categorical feature to SalePrice sorted by average of Saleprice by neighborhood
````Python
sns.stripplot(x = data.Neighborhood.values, y = y.values,
              order = data.groupby('Neighborhood')['SalePrice'].mean().sort_values().index,
              jitter=0.1, # randomly shift the points horizontally to improve visibility
              alpha=0.5)
plt.xticks(rotation=45)
````
- **pointplot**  
This is similar to above but in a different format
````Python
sns.pointplot(x = data.Neighborhood.values, y = y.values,
              order = data.groupby('Neighborhood')['SalePrice'].mean().sort_values().index)
plt.xticks(rotation=45)
````
To change the size of a seaborn plot:  
`sns.set(rc={'figure.figsize':(20,10)})  # for matplotlib use plt.figure(figsize=(20,10))`  

# Preparing the Data #
**Any modification applied on train set must be applied to test set as well.**
````Python
predictors = ['LotArea', 'OverallQual', 'OverallCond', 'YearBuilt', 'YearRemodAdd', 'TotalBsmtSF', 'BedroomAbvGr', 'GarageArea']
X = data[predictors]
y = np.log(data.SalePrice) # if it is skewed
predictions = np.exp(model.predict(test_data))

from sklearn.model_selection import train_test_split
X_train, X_val, y_train, y_val = train_test_split(X, y, random_state = 0)  # splitting data into training and validation data

data.dropna(axis=0, subset=['SalePrice'], inplace=True)  # deleting rows that contain missing entry in SalePrice column
reduced_data = data.drop(specific_columns, axis=1)  # dropping specific columns
data = data[data['LotFrontage'] < 300]  # removing some rows
````
### Imputation ###
Imputation must be done after splitting data.
````Python
from sklearn.impute import SimpleImputer
imp = SimpleImputer()
data_with_imputed_values = pd.DataFrame(imp.fit_transform(data)) 
# in the previous line casting to Pandas DataFrame will lose the column titles. To add titles back:
data_with_imputed_values.columns = data.columns  # 'columns' attribute refers to column labels of DataFrame
# Alternatively we can do:
data_with_imputed_values = pd.DataFrame(imp.fit_transform(data), columns=data.columns) 
````

### one_hot_encoding columns with non-numeric values ###
We want to choose those non-numeric columns that don't have too many categories.
````Python
initial_train = data.drop(['Id', 'SalePrice'], axis=1)
low_cardinality_cols = [ cname for cname in initial_train.columns 
                        if initial_train[cname].dtype == "object"
                        and initial_train[cname].nunique() < 10 ]
numeric_cols = [ cname for cname in initial_train.columns
                if initial_train[cname].dtype in ['int64', 'float64'] ]

train_predictors = low_cardinality_cols + numeric_cols
X = initial_train[train_predictors]

one_hot_encoded_X = pd.get_dummies(X)
...
# since one-hot-encoding produces new columns, we need to aligh training and test data
_, aligned_test = one_hot_encoded_X.align(one_hot_encoded_test, join='left', axis=1, fill_value=0)
````
### Manual one_hot_encoding ###
We can manually partition categories and assign values to each group
````Python
def encode(x): 
    return 1 if x == 'Partial' else 0
data['enc_condition'] = data.SaleCondition.apply(encode)
test['enc_condition'] = data.SaleCondition.apply(encode)
````
# Building Models #
### Decision Tree Model ###
Decision trees for classification: divide data based on different features and choose the one that results in the least impurity. Repeat this for next level. This is a greedy algorithm.
````Python
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_absolute_error

decisionTree_model = DecisionTreeRegressor()  # Define model
decisionTree_model.fit(X_train, y_train)  # Fit model
print(mean_absolute_error(y_val, decisionTree_model.predict(X_val)))  # calculating MAE of predicted 'SalePrice'

# trying different max_leaf_nodes for the decision tree
# Note that a deep tree with lots of leaves will overfit because each prediction is coming from historical data from only 
# the few houses at its leaf. But a shallow tree with few leaves will perform poorly because it fails to capture as many 
# distinctions in the raw data.
def get_mae(max_leaf_nodes, predictors_train, predictors_val, targ_train, targ_val):
    model = DecisionTreeRegressor(max_leaf_nodes=max_leaf_nodes, random_state=0)
    model.fit(predictors_train, targ_train)
    return mean_absolute_error(targ_val, model.predict(predictors_val))

for max_leaf_nodes in [5, 50, 500, 5000]:
    mae = get_mae(max_leaf_nodes, X_train, X_val, y_train, y_val)
    print("Max leaf nodes: {} \t\t Mean Absolute Error: {}".format(max_leaf_nodes, mae))
````
### Random Forest Model ###  
````Python
from sklearn.ensemble import RandomForestRegressor

forest_model = RandomForestRegressor()
forest_model.fit(X_train, y_train)
print(mean_absolute_error(forest_model.predict(X_val), y_val))
````
### Pipelines ###
````Python
from sklearn.pipeline import make_pipeline

my_pipeline = make_pipeline(Imputer(), GradientBoostingRegressor(n_estimators=1000, learning_rate=0.05, random_state=0))
#print(my_pipeline)
my_pipeline.fit(X_train, y_train)
print("pipeline mae error: ", mean_absolute_error(my_pipeline.predict(X_val), y_val), "\n")

_, aligned_test = one_hot_encoded_X.align(one_hot_encoded_test, join='left', axis=1, fill_value=0)
predicted_prices = my_pipeline.predict(aligned_test)
print("using pipeline", predicted_prices)
````
### cross-validation ###
Cross-validation is not for improving model-fitting (by using more data). It is for comparing models and tuning model parameters.  
Cross-validation is good for smaller datasets (on larger datasets it takes way more time; besides, in larger datasets we 
have enough data for train and validation, so no need for cross-validation).  
If different experiments (folds) of cross-validation give similar result, then train_test_split is probably enough.
````Python
my_pipeline = make_pipeline(Imputer(), GradientBoostingRegressor(n_estimators=1000, learning_rate=0.05, random_state=0))

from sklearn.model_selection import cross_val_score
# kfold has shuffle option but cross_val_score doesn't. We can shuffle manually using sklearn.utils.shuffle:
# from sklearn.utils import shuffle
# data = shuffle(data)
scores = cross_val_score(my_pipeline, one_hot_encoded_X, y, 
                         scoring='neg_mean_absolute_error', 
                         cv=5)  # number of folds
print(scores)
#Because of the convention that higher return values are better than lower ones, we had the scorer 'neg_mean_absolute_error' so here we multiply by -1.
print('MAE (average among k-folds): ', (-1 * scores.mean()))
````

### XGBoost models ###
- The hyperparameters that have the greatest effect on XGBoost objective metrics are: `alpha`, `min_child_weight`, `subsample`, `eta`, and `num_round` ([all parameters](https://xgboost.readthedocs.io/en/latest/parameter.html)).  
- Both Randomforest and XGBoost models use tree ensembles. But Boosting is based on weak learners (high bias, low variance), while Random Forest uses fully grown decision trees (low bias, high variance). So Boosting is trying to reduce bias while Random Forest tries to reduce variance.
````Python
from xgboost import XGBRegressor

pipeline = make_pipeline(SimpleImputer(), XGBRegressor(n_estimators=10000, learning_rate=0.01, random_state=0, n_jobs=4, reg_alpha=1, 
                                                       xgbregressor__early_stopping_rounds=20, xgbregressor__eval_set=[(X_val, y_val)], xgbregressor__verbose= False))
pipeline.fit(X_train, y_train)
predictions = np.exp(pipeline.predict(X_val))
actual_values = np.exp(y_val)
print("RMSLE: ", sqrt(mean_squared_log_error(predictions, actual_values)))
````
## Analyzing Data After Model Fitting ##
### Feature Importance
````Python
regressor = tree.DecisionTreeRegressor()
regressor.fit(x, y)
print(regressor.feature_importances_)
# print names together with the values
feature_importance = []
for name, importance in zip(X, regressor.feature_importances_):
  feature_importance.append((name, importance))
feature_importance.sort(key=lambda x:-x[1])
print(*feature_importance, sep='\n')
````
### Dependence plots ###
````Python
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.ensemble.partial_dependence import partial_dependence, plot_partial_dependence

GBR_model = GradientBoostingRegressor(n_estimators=1000, learning_rate=0.05, random_state=0)
GBR_model.fit(X, y)

fig, my_plots = plot_partial_dependence(GBR_model, # apparently it works only with GradientBoosting models
                                features=[0, 1, 2], # column numbers of plots we want to show
                                X=X,  # raw predictors data.
                                feature_names=['a', 'b', 'c'], # labels on graphs
                                grid_resolution=10, # number of values to plot on x axis 
                                n_cols=2) # number of columns for showing the data on the screen
fig.show()
````
### plot_importance ###
````Python
from xgboost import plot_importance
X_train, X_val, y_train, y_val = train_test_split(imputed_one_hot_encoded_X, y, test_size=0.2, shuffle=True, random_state=3)

d_train = xgboost.DMatrix(X_train, label=y_train)
d_val = xgboost.DMatrix(X_val, label=y_val)

params = { # by default it does linear regression. 'objective' defines it.
    "eval_metric": "mae",
    "base_score": np.mean(y_train),
    "learning_rate":.01  # AKA 'eta'
}

model = xgboost.train(params, d_train, evals = [(d_val, "val")], verbose_eval=10, num_boost_round=1000, early_stopping_rounds=100)

xgboost.plot_importance(model, max_num_features=20, height=.5)
plt.title("importance_type: weight") # options for importance_type: weight, cover, gain
plt.show()
````
### SHAP Value Analysis ###  
measures how much each feature in our model contributes
````Python
import shap
shap_values = shap.TreeExplainer(model).shap_values(one_hot_encoded_X)
global_shap_vals = np.abs(shap_values).mean(0)[:-1]
inds = np.argsort(global_shap_vals)
inds = inds[-20:]
y_pos = np.arange(20)
plt.barh(y_pos, global_shap_vals[inds], color="#1E88E5")
plt.yticks(y_pos, one_hot_encoded_X.columns[inds])
plt.gca().spines['right'].set_visible(False)
plt.gca().spines['top'].set_visible(False)
plt.xlabel("mean SHAP value magnitude (change in log odds)")
plt.gcf().set_size_inches(12, 9)
plt.show()
````

# Preparing Predicted Data for Submitting #
````Python
_, aligned_test = one_hot_encoded_X.align(one_hot_encoded_test, join='left', axis=1, fill_value=0)
predicted_prices = my_pipeline.predict(aligned_test)  # if np.log was done, here we need to do np.exp

my_submission = pd.DataFrame({'Id': test_data.Id, 'SalePrice': predicted_prices})
my_submission.to_csv('submission12.csv', index=False)
````

# Improvements
````Python
import seaborn as sns

data_train.Foundation.value_counts() # non-visual
sns.countplot(data_train.Foundation) # visual
````
- After checking  the test data for the same column, if the distribution is similar, merge small categories together
- Decision trees, for example, tend to attach less importance to sparse features and, as a result, dummy-encoded variables may be ignored in favour of their numerical counterparts, causing model performance to suffer.
- `MSSubclass` is categorical in disguise
