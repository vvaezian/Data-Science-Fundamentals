#### Styles
```python
plt.style.available  # lists all styles
plt.style.use('fivethirtyeight')
```

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
```
Plotting multiple dataframes in one plot, but not on top of each other
```python
df.plot(subplots=True, sharex=True, sharey=False, layout=(2,2))
```
or
```python
# minimal
fig, axs = plt.subplots()
df1.plot(ax=ax)
df2.plot(ax=ax)

# more options
fig, axs = plt.subplots(2, 1, figsize=(5,10))
df1.plot(x='col_x1', y='col_y1', ax=axs[0])
df2.plot(x='col_x2', y='col_y2', ax=axs[1])
```
Alternatively we can concatenate the df's and then plot.

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

#### CountPlot
```python
# Demacrat-republican classification example
# since all the features in this example are binary, countplot works better
plt.figure()
sns.countplot(x='predictor_col', hue='target_col', data=df, palette='RdBu') # RdBU means: red blue
plt.xticks([0, 1], ['No', 'Yes'])  # mapping 0, 1 to 'No', 'Yes'
plt.show()
```
### Colormap (useful for timeseries)
```python
df.plot.scatter('A', 'B', c=df.index, cmap=plt.cm.viridis, colorbar=False)  # it doesn't have to be scatterplot
```
