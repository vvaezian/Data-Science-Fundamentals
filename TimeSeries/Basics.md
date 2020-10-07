- If the index of a dataframe is DatetimeIndex, the dataframe is a timeseries.
- Specifying the index:
  - While reading the csv `stock_1_2 = pd.read_csv('s1_2.csv', parse_dates=['date'], index_col='date')`
  - If date is part of the data as a column, we can use `set_index`: `df.set_index('index_col', inplcae=True)`.  
  If the type of the date column is string we need to convert it first: `df.date = pd.to_datetime(df.date)`
  - `reindex` replaces the current index with the given index
  ```python
  new_index = pd.date_range(start='2020', periods=12, freq='M')
  df.reindex(new_index)  # there is a 'method' option with possible values {None, ‘backfill’/’bfill’, ‘pad’/’ffill’, ‘nearest’}
                         # which determines how the null values should be treated (default None, i.e. don't change them)
  ```
```python
import pandas as pd
from datetime import datetime

time_stamp = pd.Timestamp('2020-06-12')  # equivalent of pd.Timestamp(datetime(2020, 06, 12))
# pd.Timestamp(year = 2020, month = 06, day = 12, hour = 10, second = 49, tz = 'US/Central') 
time_stamp.year # .month .week .day
time_stamp.day_name()  # Friday
time_stamp.weekday()  # 4  (starts from Monday=0)

period = pd.Period('2020-06', 'M')  # If we omit 'M', it defaults to it. The defaul frequency for '2020-06-01' is day
                                    # M:'month end', MS:'month start', BM: 'business month end', BMS: 'business month start'
period.asfreq('D')  # Period('2020-06-30', 'D')
period.to_timestamp()  # Timestamp('2020-06-01 00:00:00')
time_stamp.to_period('M')  # Period('2020-06', 'M')
period + 1  # Period('2020-07', 'M')
# timestamp can also have frequency
ts = pd.Timestamp('2020-06-12', 'M')
ts + (2 * ts.freq)  # Timestamp('2020-07-31 00:00:00', freq='M')
```
#### Sequence of dates and times
```python
index = pd.date_range('2020-06-12', periods=3, freq='D')  # DatetimeIndex(['2020-06-12', '2020-06-13', '2020-06-14'], dtype='datetime64[ns]', freq='D')
index[0]  # Timestamp('2020-06-12 00:00:00', freq='D')
period_index = index.to_period()  # PeriodIndex(['2020-06-12', '2020-06-13', '2020-06-14'], dtype='period[D]', freq='D')
period[0]  # Period('2020-06-12', 'D')

data = np.random.random((3,2))  # array of random numbers, 3 rows, 2 columns
df = pd.DataFrame(data, index=index)  
# if index is part of the data as a column, we use set_index: df.set_index('index_col', inplcae=True)
# if date column doesn't have the datetime type, we can convert it: data.date = pd.to_datetime(data.date)
 	              0 	      1
2020-06-12 	0.160132 	0.500385
2020-06-13 	0.287287 	0.539113
2020-06-14 	0.749582 	0.899299
# We can indicate the index when loading from CSV
data = pd.read_csv('data.csv', parse_dates=['date_col'], index_col='date_col')

import matplotlib.pyplot as plt
data[data.columns[-1]].plot()  # one column
data.plot(subplots=True)  # all columns
# plt.tight_layout()
plt.show()
```

#### Partial Index
```python
# *** use string, not int ***
data['2020']  # all rows that their time is in 2020
data['2020-01': '2020-06']  # inclusive
data.loc['2020-04-01', 'target_col']  # specific value. use .reset_index(drop=True) to select column with the default numerical index
data.asfreq('D')  # if data doesn't have freq, we can add it using asfreq().
                  # this cause NaN values for the dates that the index doesn't cover
                  # 'D' is for calendar days. 'B' is for business days

data['shifted'] = data.target_col.shift(periods=1)  # copies the data from previous line (periods=1 is default)
data['change'] = data.target_col.div(data.shifted)
data['pct_change'] = data.change.sub(1).mul(100)  # subtract by 1 any multiply by 100. e.g. if change is 1.1, this means 10% increase.
data['pct_change2'] = data.target_col.pct_change().mul(100)  # equivalent of previous operations
data['diff'] = data.target_col.diff()  # the difference between the value of the cell in the current row and its value in the previous row
# all these methos have the attribute 'period' which determines how many lines back or forward to look.
```
when we have multiple columns, to normalize data, we divide values by the value of the first row and multiply by 100
so all columns start with 100
```python
prices = pd.read_csv('asset.csv', parse_dates=['DATE'], index_col='DATE')
first_row_prices = prices.iloc[0]
normalized = prices.div(first_row_prices).mul(100)
```
#### Compare the performance of two things with a benchmark
```python
stock_1_2 = pd.read_csv('s1_2.csv', parse_dates=['date'], index_col='date')
stock_3 = pd.read_csv('s3.csv', parse_dates=['date'], index_col='date')
data = pd.concat([stock_1_2, stock_3], axis=1).dropna()
normalized = data.div(data.iloc[0]).mul(100)  # normalize data
normalized['col1', 'col2'].sub(normalized['col3'], axis=0)  # subtract col3 from the two columns
```
#### Up-sampling/down-sampling
use resample, or asfreq
for upsample we need to fill out the null values. `.ffill()`, `.bfill()`, ...  
for downsample we need to determine how the new values are defined. `.mean()`, `.first()`, `.last()`, ...  
  
reindex: replace the current index with the given index
```python
new_index = pd.date_range(start='2020', periods=12, freq='M')
df.reindex(new_index)  # there is a 'method' option with possible values {None, ‘backfill’/’bfill’, ‘pad’/’ffill’, ‘nearest’}
                       # which determines how the null values should be treated (default None, i.e. don't change them)
```
resample is more general version of asfreq. It is like groupby 
```python
df.resampl(rule='M', how='last')  # or df.resample('M').last()
ts.resample('MS').ffill().add_suffix()  # add_suffix adds a suffix to the column name
ts.resample('MS').interpolate()
ts.resample('MS').mean()  # if we do ts.asfreq().mean() it outputs the mean of the whole column, not week-by-week as in resample('W')
ts.resample('M').agg(['mean', 'median', 'std'])  # creating multiple columns for different measures
# .ffil(), .intepolate() and other relevant methods can be applied to any time series data, we don't have to use resample() first.
# if applied to a dataframe, it gets applied to all columns
```

### Window functions 
```python
# .rolling()
# moving average:
data.col.rolling(window='30D').mean()  # .rolling has the 'min_periods' option: Minimum number of observations in window required to have a value
q10 = data.col.rolling.quantile(0.1).to_frame('q10')  # if we are adding to a df, we don't need the to_frame() call (e.g. data['q10'] = ...
median = rolling.median().to_frame('median')
q90 = data.col.rolling.quantile(0.9).to_frame('q90')
pd.concat(['q10','median','q90'], axis=1).plot()

# .expanding 
df.col.expanding.sum()  # same as df.col.cumsum()
df.col.expanding.max()  # running maximum
```

#### Example: cumulative return
single period return r_t: current price over last price minus 1 [r_t = p_t/p_{t-1}) - 1]  
multi period return: product of (1 + r_t) for all  periods minus 1 [(1 + r_1) * ... * (1 + r_T) - 1]
```python
returns = data.pct_change()
returns_plus_one = returns.add(1)  # -0.005 pct_change becomes 0.995 
cumulative_return = returns_plus_one.cumprod().sub(1)  # multiplying rows up to the current row
```
#### Example: cumulative for rolling 1-year period
```python
def multi_period_return(period_returns):
    return np.prod(period_returns + 1) - 1
daily_returns = data.pct_change()
rolling_annual_returns = daily_returns.rolling('360D').apply(multi_period_return)
rolling_annual_returns.mul(100).plot()
```
#### total return
```python
data.iloc[-1].div(data.iloc[0]).sub(1).mul(100)
```
