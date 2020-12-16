We can access the cross-validation indeces using `.split()`:
```py
from sklearn.model_selection import KFold
cv = KFold(n_splits=10)

for train_indeces, test_indeces in cv.split(X, y):
    model.fit(X[train], y[tr])
```
