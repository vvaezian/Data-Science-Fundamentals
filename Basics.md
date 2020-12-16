We can access the cross-validation indeces using `.split()`:
```py
from sklearn.model_selection import KFold
cv = KFold(n_splits=10)

for train_indeces, test_indeces in cv.split(X):
    X_train, X_test = X[train_indeces], X[test_indeces]
    y_train, y_test = y[train_indeces], y[test_indeces]
```
