```python
ax = df.plot(figsize=(12,6), fontsize=12, linewidth=3, linestyle='--', color='blue')
ax.set_xlabel('Date', fontsize=15)
ax.set_ylabel('Price')
ax.set_title('My title')
plt.show()

ax.axvline('1939-01-01', color='red', linestyle='--')  # vertical line
ax.axhline(4, color='green', linestyle='--')  # horizontal line
ax.axvspan('1900-01-01', '1915-01-01', color='red', alpha=0.3)  # vertical region
ax.axhspan(6, 8, color='green', alpha=0.3)  # horizontal region

df.plot(subplots=True, sharex=True, sharey=False)
```
#### Styles
```python
plt.style.available  # lists all styles
plt.style.use('fivethirtyeight')
```
#### Histogram
```python
df.plot(kind='hist'[, bins=10, subplots=True])  # histogram. This is better than df.hist()
# density plots are like a smoother version of histograms
df.plot(kind='density')
```
#### Boxplot (AKA box and whisker plot)  
- The box extends from the lower to upper quartile values of the data, with a line at the median. 
- Let IQR be the interquartile range (Q3-Q1). The upper whisker will extend to the biggest data point less than `Q3 + whis*IQR` (`whis=1.5` by default). Similarly, the lower whisker will extend to the smallest data point greater than `Q1 - whis*IQR`.  
Beyond the whiskers, data are considered outliers and are plotted as individual points.  
Set `whis='range'` to force the whiskers to be the min and max of the data.  
Set `whis` to an ascending sequence of percentile (e.g., `[5, 95]`) to set the whiskers at specific percentiles of the data
