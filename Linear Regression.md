AKA Ordinary Least Squares

```python
import statmodels.api as sm
sm.OLS(y, dfx).fit()  # need to add a column of 1's to dataframe using sm.add_constant(dfx)

import numpy as np
np.polyfit(x, y, deg=1)

import pandas as pd
pd.ols(y,x)

from scipy import stats
stats.linregres(x,y)
```
