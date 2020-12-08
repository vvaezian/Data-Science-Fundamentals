To more easily see the outliers, it's a good idea to transform the data. E.g. calculate percent change between the current value and mean of past n values:
```python
def percent_change(series):
    previous_values = series[:-1]
    last_value = series[-1]

    percent_change = (last_value - np.mean(previous_values)) / np.mean(previous_values)
    return percent_change

# Apply the custom function and plot
ts_pct = ts.rolling(20).apply(percent_change)
ts_pct.loc["2015":"2020"].plot()
plt.show()
```
