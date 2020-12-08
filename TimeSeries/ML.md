To more easily detect and handle the outliers, it's a good idea to transform the data. E.g. calculate percent change between the current value and mean of past n values:
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
TODO: In the following function exclude the data point itself from the calculation. E.g. in the first line we should calculate the abs diff between each timepoint and the series mean where the mean is calculated by excluding the data point.
```python
def replace_outliers(series):
    # Calculate the absolute difference of each timepoint from the series mean
    abs_diff_from_mean = np.abs(series - np.mean(series))

    # Calculate a mask for the differences that are > 3 standard deviations from the mean
    mask = abs_diff_from_mean > (np.std(series) * 3)
    
    # Replace these values with the median accross the data
    series[mask] = np.nanmedian(series)
    return series

# Apply the function to the timeseries and plot the results
ts_pct = ts_pct.apply(replace_outliers)
ts_pct.loc["2015":"2020"].plot()
plt.show()
```
