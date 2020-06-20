- To start Spark shell (not the pySpark): `spark-shell`
- To start the pySpark shell: `pySpark`
- Run pyspark file from Powershell (doesn't work in Git-Bash): `spark-submit .\pySpark_test.py`

Example from `examples\src\main\python`
```python
from random import random
from operator import add

from pyspark.sql import SparkSession

spark = SparkSession\
    .builder\
    .appName("PythonPi")\
    .getOrCreate()

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
Found this online:
```python
from pyspark import SparkContext, SparkConf

conf = SparkConf().setAppName("Montecarlo_PI")
sc = SparkContext(conf=conf)

import random
NUM_SAMPLES = 100000000
def inside(p):
  x, y = random.random(), random.random()
  return x*x + y*y < 1

count = sc.parallelize(range(0, NUM_SAMPLES)).filter(inside).count()
pi = 4 * count / NUM_SAMPLES
print("Pi is roughly", pi)
```

