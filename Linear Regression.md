AKA Ordinary Least Squares

```python
import statmodels.api as sm
sm.OLS(y,x).fit()

import numpy as np
np.polyfit(x, y, deg=1)

import pandas as pd
pd.ols(y,x)

from scipy import stats
stats.linregres(x,y)
```
