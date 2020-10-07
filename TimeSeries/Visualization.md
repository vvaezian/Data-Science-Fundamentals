```python
plt.style.available  # lists all styles
plt.style.use('fivethirtyeight')

ax = df.plot(figsize=(12,6), fontsize=12, linewidth=3, linestyle='--', color='blue')
ax.set_xlabel('Date', fontsize=15)
ax.set_ylabel('Price')
ax.set_title('My title')
plt.show()
```
