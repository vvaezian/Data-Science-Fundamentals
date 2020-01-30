### Preliminary steps
````python
# Innitialize
import h2o
h2o.init(max_mem_size="6G")  # if max_mem_size is not provided it uses 1G for 32 bit Java and 1/4 available system memory in 64 bit Java. 
                             # (this info is from https://h2o-release.s3.amazonaws.com/h2o/rel-wheeler/4/docs-website/h2o-docs/booklets/RBooklet.pdf 
                             # which is for R. On my system with 16 GB RAM it value was 3.979. 
                             # So it seems it uses 1/4 total memory (given that amount is free), not the available memory (at least in Python).

# Import data into DataFrame
data = h2o.import_file('file.csv', col_types={'myCol':'enum'})

# Check column types
data.types

# Create DataFrom directly
df_direct = h2o.H2OFrame({'a':[1, 12, 33], 'b':[40, 15, 26], 'c':[7, 8, 9]})

# Create DataFrame from Pandas DataFrame
df_from_Pandas = h2o.H2OFrame(pdFrame, column_types={'myCol':'enum'})
````

### some preliminary functions
````Python
help(h2o.import_file)  # Getting help on functions
data.names  # same as `data.columns`
data.head() # to show more than 10 rows use `rows=n`
data.describe()
````

### Preparing the data
````Python
# Checking if target variable is skewed
y = data[data.names[-1]]
y.hist(100)  # y.skewness() gives a numerical measure of skewness

# Say we see data is skewed. So we log the target variable and attach it to the rest of data
yLog = y.log()
X = data[data.names[:-1]]
data_unskewed = X.cbind(yLog) # combining two dataframes  
  
df1.rbind(df2)  # vertically adds the two df's (df's need to have same columns and types)  
df1.cbind(df2)  # horizontally adds the two df's

# Unlike Sklearn, in H2O we don't need to separate predictors from target variable for splitting data
train, val, test = data_unskewed.split_frame([.6, .2], seed = 10)

# to split manually:
r_nums = data.runif()  # .runif assigns a random number to each row, from a uniform set of random numbers between 0 and 1. 
                       # We use this for a shuffled split of data
train = data[r_nums  < 0.6]
val = data[(r_nums >= 0.6) & (r_nums < 0.8)]
test = data[r_nums  >= 0.8]
````
### Building a Model
````Python
from h2o.estimators.random_forest import H2ORandomForestEstimator
rf_model = H2ORandomForestEstimator()
rf_model.train(x=train.names[:-1], y=train.names[-1], training_frame=train, validation_frame=val)  # set `model_id='model_name'` to be able to set the name when saving the model
````
For **cross-validation**, set `nfolds` to a number n greater than 1. It builes `n+1` models, the first n, on 1/n of data, and then a final one on the whole dataset which can be used for prediction.

### After Build
````Python
# variable importance
rf_model.varimp()  # headers: variable, relative_importance, scaled_importance, percentage

  # Excluding the fisrt 4 columns from variable importance and rescaling:
  mmin = final_model.varimp()[-1][2]
  mmax = final_model.varimp()[4][2]
  total_points = sum([i[1] for i in final_model.varimp()[4:]])

  demog_varimps = [[i[0], (i[2]-mmin)/(mmax-mmin), i[1]/total_points] for i in final_model.varimp()[4:]]
  demog_varimps

#Get performance results of the model
rf_model.model_performance(test_data=test)

#To get only a specific metric  
rf_model.model_performance(test_data=test)['mae'] # or rf_model.mae(test_data=test)
  # to see how good this mae is, we can compare it with the case if we use mean of target variable as the prediction:
  target = data['target_col']
  m = target.mean()[0]  # .mean() returns a list, so we use [0] to get the value
  mean_mae = abs(target-m).sum()/len(target)
  
m = target.mean()[0]
# Get the predicted Values
rf_model.predict(test)

# Save the model
h2o.save_model(model=my_model)  # need to have the model_id attribute set before saving.
                                # We can do it either when training the model, or later `my_model.model_id = '...'`

# Load the model
model = h2o.load_model('C:\\PATH\\TO\\SavedModel')

# Export dataframe to file
h2o.export_file(df, 'C:\\PATH\\TO\\FILE')

# Shutdown the Cluster
h2o.cluster().shutdown()
````
