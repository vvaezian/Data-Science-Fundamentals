### Bayes' Rule
- The opposite of FP is TN (not TP)
- The FP is the accuracy when test is applied only to negative population.
- It may be easier to see the Bayes rule for `P(have_disease | test_positive)` as the conditional probability rule. We just need to note that `P(have_disease and test_positive)` (the numerator) is not just `P(TP)`, but (because of the point above) is `P(TP) * P(positive_population)`. Similarly for the denominator (`P(test_positive)`) we have `P(TP) * P(positive_population) + P(FP) * P(negative_population)`.

### Mean Absolute Error (MAE) vs. Root mean squared error (RMSE)
- RMSE is more sensitive to outliers.
Since the errors are squared before they are averaged, the RMSE gives a relatively high weight to large errors. 
This means the RMSE should be more useful when large errors are particularly undesirable.
So if being off by 10 is more than twice as bad as being off by 5, RMSE should be used. If it is almost as bad, MAE should be used.
- RMSE is better than MAE when the error distribution is expected to be Gaussian.

### Bias-Variance Tradeoff
- Bias is the difference between the model's expected predictions and the true values. Variance refers to the algorithm's sensitivity to specific sets of training data.  
- High bias causes underfitting and high variance causes overfitting.  
- Ideally we need a model that has low bias and variance, but this seems not possible because reducing one increases the other.

### Accuracy, Recall, Precision, F1 ([source](https://towardsdatascience.com/beyond-accuracy-precision-and-recall-3da06bea9f6c))
- Accuracy: `# of true / (# of true + # of false)`.
- Recall: `# of true positives / (# of true positives + # of false negatives)`. Recall can be thought as ability of a classification model to find all the data points of interest in a dataset.
- precision: `# of true positives / (# of true positives + # of false positives)`. Precision can be thought of as ability of a classification model to identify only the relevant data points.
- F1: Harmonic mean of precision and recal; `2 * [(precision * recall)/(precision + recal)]`. We use the harmonic mean instead of a simple average because it punishes extreme values (if one recal or precision is very high, the other one has to be very low. This makes the F1 score very low).

Suppose we want to identify terrorists trying to board flights.  
In *imbalanced classification problems*, accuracy is not helpful. A model that marks everyone as not terrorist gives a very high accuracy (but 0 recall and 0 precision).  
There is a trade-off between recall and precision: A model that marks everyone as terrorist has perfect recal (=1.0) but very low precision.  
The F1 score gives equal weight to both measures and is a specific example of the general Fβ metric where β can be adjusted to give more weight to either recall or precision.

Visualizing Recall and precision
- Confusion matrix: shows the actual and predicted labels from a classification problem
- Receiver operating characteristic (ROC) curve: plots the true positive rate (TPR) versus the false positive rate (FPR) as a function of the model’s threshold for classifying a positive
- Area under the curve (AUC): metric to calculate the overall performance of a classification model based on area under the ROC curve

### Ensemble, bagging, boosting
- Ensemble: Any method that uses multipe models for the task in hand. Example: Bagging, Boosting.
- Bagging (Bootstrap Aggregating): Train multiple models *in parallel* (based on a porions of the data) and use majority vote (classification) or average (regression) of output of these models.  
  - Example: Random Forest.  
  - The idea is that averaging reduces variance and leaves bias unchanged.
  - In randmon Forests if there are M input variables, m<<M variables are selected at random and the best split on these m is used to split the node. Increasing m on one hand insreases the correlation between any two trees in the forest which increases the total error, on the other hand decreases the error of each individual tree which decrease the total error. This is the only adjustable parameter.
- Boosting: Train multiple models *sequentially* (each one based on previous one) and then adding up all these models. 
  - Example: Gradient Boosting, AdaBoost. 
  - The difference between gradient boosting and gradient descent is that "training a NN using gradient descent tweaks model parameters whereas training a GBM tweaks (boosts) the model output".

### Parameters
- **Decision Tree Learning Rate** reduces the influence of each individual tree and leaves space for future trees to improve the model

