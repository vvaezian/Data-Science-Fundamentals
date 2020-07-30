import pandas as pd
from datetime import datetime

time_stamp = pd.Timestamp('2020-06-12')  # equivalent of pd.Timestamp(datetime(2020, 06, 12))
# pd.Timestamp(year = 2020, month = 06, day = 12, hour = 10, second = 49, tz = 'US/Central') 
time_stamp.year # .month .week .day
time_stamp.day_name()  # Friday
time_stamp.weekday()  # 4  (starts from Monday=0)

period = pd.Period('2020-06', 'M')  # M (month) is the default, so we can skip it
