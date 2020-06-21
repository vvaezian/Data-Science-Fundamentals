- To start Spark shell (not the pySpark): `spark-shell`
- To start the pySpark shell: `pySpark`
- Run pyspark file from Powershell (doesn't work in Git-Bash): `spark-submit .\pySpark_test.py`

```python
from pyspark.sql import SparkSession

spark = SparkSession.builder.appName("PythonPi").getOrCreate()  # create a SparkSession
print(spark.catalog.listTables())  # list all the data inside the cluster. 

```

- Spark DataFrame is an abstraction on top of RDD's which are hard to work with directly.
- **SparkSession:** Provides a single point of entry to interact with underlying Spark functionality and allows programming Spark with Dataframe and Dataset APIs. 
- **SparkContext:** Main entry point for Spark functionality. It's used to interact with Low-Level API (create RDDs, accumulators and broadcast variables on the cluster). Before Spark 2, we had to create different contexts for working with different APIs (SQL, HIVE, Streamin). But now we can get SparkContext from SparkSession.

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
