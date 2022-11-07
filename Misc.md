#### Condifence Interval
```py
from scipy import stats

a = range(10)
print(np.mean(a))
stats.t.interval(alpha=0.95,  # the probability that a drawn random sample will be inside the returned interval
                 df=len(a) - 1,  # degrees of freedom
                 loc=np.mean(a), 
                 scale=stats.sem(a)  # sem: standard error of the mean
                 )
```
