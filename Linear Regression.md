AKA Ordinary Least Squares

```python
import numpy as np  
np.polyfit(x, y, deg=3)  # deprecated in favor of the next one

from numpy.polynomial import Polynomial
Polynomial.fit(X, Y, deg=3)

from scipy import stats
stats.linregress(x,y)

import statmodels.api as sm
sm.OLS(y, dfx).fit()  # need to add a column of 1's to dataframe using sm.add_constant(dfx)

import pandas as pd
pd.ols(y,x)
```
