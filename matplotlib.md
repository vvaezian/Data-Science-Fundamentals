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
Styles
```python
plt.style.available  # lists all styles
plt.style.use('fivethirtyeight')
```
Histogram
```python
df.plot(kind='hist'[, bins=10, subplots=True])  # histogram. This is better than df.hist()
```
