- The 3-sigma method should be used when the distribution looks gaussian and the number of outliers is low. For high number of otherwise use the Modified Thomson Tau test.
- For 3-sigma methid consider excluding the outlier candidate before calculating the 3-sigma range.  
In small datasets where many of the data points are exactly the same (e.g. price of a product doesn't change often) 
it may happen that in the subset of data points which we want to do the calculation, all the data point are the same (e.g. all the same price). 
This causes the sigma (and so 3xsigma) to be zero and therefore any different data point no matter how close to these numbers would be regarded as an outlier.
To remedy this, we can add a noise to data points.
