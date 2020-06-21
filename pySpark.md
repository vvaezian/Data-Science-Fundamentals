- To start Spark shell (not the pySpark): `spark-shell`
- To start the pySpark shell: `pySpark`
- Run pyspark file from Powershell (doesn't work in Git-Bash): `spark-submit .\pySpark_test.py`
- Spark DataFrame is an abstraction on top of RDD's which are hard to work with directly.
- **SparkSession:** Provides a single point of entry to interact with underlying Spark functionality and allows programming Spark with Dataframe and Dataset APIs. 
- **SparkContext:** Main entry point for Spark functionality. It's used to interact with Low-Level API (create RDDs, accumulators and broadcast variables on the cluster). Before Spark 2, we had to create different contexts for working with different APIs (SQL, HIVE, Streamin). But now we can get SparkContext from SparkSession.

```python
from pyspark.sql import SparkSession

spark = SparkSession.builder.appName("PythonPi").getOrCreate()  # create a SparkSession
print(spark.catalog.listTables())  # list all the data inside the cluster. 

query = "SELECT * FROM myTable limit 100"
spark_df = spark.sql(query)
spark_df.show()

# When the heavy-lifting is done with Spark we can transform the DataFrame to a Pandas DataFrame to explore the data easier.
pd_df = spark_df.toPandas()
pd_df.head()

# from pd to spark is possible as well
spark_df = spark.createDataFrame(pd_df)  # stored locally (?)
spark_df.createOrReplaceTempView("temp_table_name")  # stored on the cluster. 
                                                     # Can only be accessed from the current session
                                                       

# csv to spark_df
spark_df = spark.read.csv(file_path, header=True)
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
