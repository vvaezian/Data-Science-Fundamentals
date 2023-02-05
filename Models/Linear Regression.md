AKA Ordinary Least Squares

```python
import numpy as np  
np.polyfit(x, y, deg=3)  # deprecated in favor of the next one

from numpy.polynomial import Polynomial
p = Polynomial.fit(X, Y, deg=3, domain=[X[0], X[-1]])

from scipy import stats
stats.linregress(x,y)

import statmodels.api as sm
sm.OLS(y, dfx).fit()  # need to add a column of 1's to dataframe using sm.add_constant(dfx)

import pandas as pd
pd.ols(y,x)
```
