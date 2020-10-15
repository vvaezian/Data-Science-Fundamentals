- After fitting the model, the next value of the time series can be predicted by replacing the coffecients in the formula.  
For example for an AR(1) model we have `y_t = a_1 * y_{t-1} + e_t`. Suppose the current value is 10 and the coefficient from fitting the model (here, the lag1 coefficient) is .8 and the standard deviation of error ('std err' in the summary) is 1. Then for the next value x we have: `0.8 * 10 - 1 < x < 0.8 * 10 + 1`

```python
from statsmodels.tsa.arima_model import ARMA

# Fit an AR(1) model to the simulated data
mod = ARMA(data, order=(1,0))  # data can be df, series or Numpy array
res = mod.fit()
print(res.summary())
print(res.params)  # returns μ and φ
print(res.aic)
print(res.bic)

# prediction on current data
forecast = res.get_prediction(start = -10) # how many steps back to start the forecast
                                           # set `dynamic=True` to go more than one-step-ahead prediction.
mean_forecast = forecast.predicted_mean
confidence_intervals = forecast.conf_int()  # a df with lower and upper limits

# future forecasts
forecast = res.get_forecast(steps=10) 
mean_forecast = forecast.predicted_mean
confidence_intervals = forecast.conf_int()  # a df with lower and upper limits

res.plot_predict(start=990, end=1010)  # if data has index we can use plot_predict(start='2020-08-01', end='2020-10-01')
plt.show()
```
 - ARMAX is for using external (AKA exogenous) variables in addition to the timeseries variables. (ARMA + linear regression)
```python
# ARMAX(1,1)
y_t = x_1 * z_t + a_1 * y_{t-1} + m_1 * e_{t-1} + e_t
```
Example: For Modellig personal productivity in the current, we may include productivity in previous days (timeseries variable) and the number of hours slept last night (external variable).
```python
model = ARMA(df['productivity'], order=(2,1), exog=df['hours_sleep'])
```
