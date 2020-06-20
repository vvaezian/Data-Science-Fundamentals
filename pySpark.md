- To start Spark shell (not the pySpark): `spark-shell`
- To start the pySpark shell: `pySpark`
- Run pyspark file from Powershell (doesn't work in Git-Bash): `spark-submit .\pySpark_test.py`

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
