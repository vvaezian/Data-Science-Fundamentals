- To start Spark shell (not the pySpark): `spark-shell`
- To start the pySpark shell: `pySpark`
- Run pyspark file from Powershell (doesn't work in Git-Bash): `spark-submit .\pySpark_test.py`
- Spark DataFrame is an abstraction on top of RDDs which are hard to work with directly.
- **SparkSession:** Provides a single point of entry to interact with underlying Spark functionality and allows programming Spark with Dataframe and Dataset APIs. 
- **SparkContext:** Main entry point for Spark functionality. It's used to interact with Low-Level API (create RDDs, accumulators and broadcast variables on the cluster). Before Spark 2, we had to create different contexts for working with different APIs (SQL, HIVE, Streaming). But now we can get SparkContext from SparkSession.

```python
from pyspark.sql import SparkSession

spark = SparkSession.builder.appName("PythonPi").getOrCreate()  # create a SparkSession
print(spark.catalog.listTables())  # list all the data inside the cluster. 
```
Using **SQL** queries to create DataFrames
```python
query = "SELECT * FROM myTable limit 100"
df = spark.sql(query)
df.show()
```
When the heavy-lifting is done by Spark we can transform the DataFrame to a **Pandas** DataFrame to explore the data easier.
```python
pd_df = spark_df.toPandas()
pd_df.head()

# from pd to spark is possible as well
spark_df = spark.createDataFrame(pd_df)  # stored locally (?)
spark_df.createOrReplaceTempView("temp_table_name")  # stored on the cluster. 
                                                     # Can only be accessed from the current session
```                                                       
Import **CSV**
```python
df = spark.read.csv(file_path, header=True)
```

Spark DataFrames are **immutable**
```python
spark_df = spark.table("test_table")  # using data already in the cluster
spark_df = spark_df.withColumn("col1", spark_df.col1 - 1)  # deducting 1 from all elements of the column "col1"
spark_df = spark_df.withColumn("newCol", spark_df.col2 / 60)  # adding a new column constructed from an existing column
spark_df = spark_df.withColumnRenamed("OldColName", "NewColName")  # rename a column
```
**Filtering** the data (equivalent of "where" clause). Both of the following return the same result
```python
long_flights1 = flights.filter("distance > 1000")
long_flights2 = flights.filter(flights.distance > 1000)  # 'flights.distance > 1000' returns a boolean column
df.filter("col1 is not null and col2 is not null")
```

Difference between **`select`** and **`withColumn`** is that the latter returns the whole df while the fomer returns only the selected cols
```python
selected_cols = flights.select("origin", "dest")
selected_cols = flights.select(flights.origin, flights.dest)  
# when using the dot notation, we can do column operations as well
selected_cols = flights.select(flights.duration/60.alias("duration_hrs"), flights.dest) 
# the alias operation can be used in string notation as follows
flights.selectExpr("duration/60 as duration_hrs")
```
**Grouping**
```python
flights.groupBy("origin", "dest").avg("distance").show()
flights.groupBy(flights.origin, flights.dest).avg("distance").show()

# We can calculate standard deviation and other function on grouped data as follows
import pyspark.sql.functions as F
flights_grouped = flights.groupBy(flights.origin, flights.dest).avg("distance")  # from the last line of code above
flights_grouped.agg(F.stddev("distance")).show()
```
**Joining**  
If table1 has columns A, B and table2 has columns A, C, the following results in a table with columns A, B, C i.e. it includes the common column once
```python
joined_tables = table1.join(table2, on="common_col", how="leftouter")
```

When we import data, Spark guesses **column type**, but this is not seemless. So we need to check and `cast` to the right type.
```python
spark_df.dtypes
```

Spark only handles **numeric** data. That means all of the columns in the DataFrames must be either `integer` or `double` (decimals).
```python
df = spark_df.withColumn("col_name", spark_df.col_name.cast("integer"))
```

**One-hot Encoding**
```python
col1_indexer = StringIndexer(inputCol="col1", outputCol="col1_index")
col1_encoder = OneHotEncoder(inputCol="col1_index", outputCol="output")
```
Spark model expects all the data be in one column. So we turn each row into a **vector**:
```python
vec_assembler = VectorAssembler(inputCols=["col1", "col2", "col3"], outputCol="features")
```
**Pipeline**
```python
from  pyspark.ml import Pipeline
pipe = Pipeline(stages=[col1_indexer, col1_encoder, col2_indexer, col2_encoder, vec_assembler])
```
**Fit** and **transoform**
```python
piped_data = pipe.fit(model_data).transform(model_data)
```
**Split** Data
```python
training, test = piped_data.randomSplit([.8, .2])
```
**Logistic Regression**
```python
from pyspark.ml.classification import LogisticRegression
lr = LogisticRegression()
```
**Parameter Grid**
```python
import pyspark.ml.tuning as tune
grid = tune.ParamGridBuilder()
grid = grid.addGrid(lr.regParam, np.arange(0, .1, .01))  # regParam corresponds to lambda
grid = grid.addGrid(lr.elasticNetParam, [0, 1])  # elasticNetParam corresponds to alpha
                                                 # 0 corresponds to Ridge and 1 corresponds to Lasso
grid = grid.build()
```
**Cross-Validation**
```python
import pyspark.ml.evaluation as evals
evaluator = evals.BinaryClassificationEvaluator(metricName="areaUnderROC")

cv = tune.CrossValidator(estimator=lr,
               estimatorParamMaps=grid,
               evaluator=evaluator
               )
models = cv.fit(training)
# Extract the best model (i.e. best hyperparameters)
best_lr = models.bestModel
```
**Prediction**
```python
test_results = best_lr.transform(test)

# Evaluate the predictions (we defined evaluator as "areaUnderROC". The closer to 1, the better the model)
print(evaluator.evaluate(test_results))
```


Example from `examples\src\main\python`
```python
from random import random
from operator import add

from pyspark.sql import SparkSession

spark = SparkSession.builder.appName("PythonPi").getOrCreate()

#partitions = int(sys.argv[1]) if len(sys.argv) > 1 else 2
partitions = 2
n = 10000000 * partitions

def f(_):
    x = random() * 2 - 1
    y = random() * 2 - 1
    return 1 if x ** 2 + y ** 2 <= 1 else 0

count = spark.sparkContext.parallelize(range(1, n + 1), partitions).map(f).reduce(add)
print("Pi is roughly %f" % (4.0 * count / n))
```
