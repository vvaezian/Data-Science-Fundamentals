# If the index of a dataframe is DatetimeIndex, the the dataframe is a timeseries.

import pandas as pd
from datetime import datetime

time_stamp = pd.Timestamp('2020-06-12')  # equivalent of pd.Timestamp(datetime(2020, 06, 12))
# pd.Timestamp(year = 2020, month = 06, day = 12, hour = 10, second = 49, tz = 'US/Central') 
time_stamp.year # .month .week .day
time_stamp.day_name()  # Friday
time_stamp.weekday()  # 4  (starts from Monday=0)

period = pd.Period('2020-06', 'M')  # If we omit 'M', it defaults to it. The defaul frequency for '2020-06-01' is day
period.asfreq('D')  # Period('2020-06-30', 'D')
period.to_timestamp()  # Timestamp('2020-06-01 00:00:00')
time_stamp.to_period('M')  # Period('2020-06', 'M')
period + 1  # Period('2020-07', 'M')
# timestamp can also have frequency
ts = pd.Timestamp('2020-06-12', 'M')
ts + (2 * ts.freq)  # Timestamp('2020-07-31 00:00:00', freq='M')

# Sequence of dates and times
index = pd.date_range('2020-06-12', periods=3, freq='D')  # DatetimeIndex(['2020-06-12', '2020-06-13', '2020-06-14'], dtype='datetime64[ns]', freq='D')
index[0]  # Timestamp('2020-06-12 00:00:00', freq='D')
period_index = index.to_period()  # PeriodIndex(['2020-06-12', '2020-06-13', '2020-06-14'], dtype='period[D]', freq='D')
period[0]  # Period('2020-06-12', 'D')

data = np.random.random((3,2))  # array of random numbers, 3 rows, 2 columns
df = pd.DataFrame(data, index=index)  
# if index is part of the data as a column we use set_index: df.set_index('index_col', inplcae=True)

 	              0 	      1
2020-06-12 	0.160132 	0.500385
2020-06-13 	0.287287 	0.539113
2020-06-14 	0.749582 	0.899299

import matplotlib.pyplot as plt
df[df.columns[-1]].plot()  # one column
plt.show()
# data.plot(subplots=True)  # all columns


# partial index
data['2020']  # all rows that their time is in 2020
data['2020-01': '2020-06']  # inclusive
data.loc['2020-04-01', 'target_col']  # specific value
data.asfreq('D')  # if data doesn't have freq, we can add it using asfreq().
                  # this cause NaN values for the dates that the index doesn't cover
                  # 'D' is for calendar days. 'B' is for business days
