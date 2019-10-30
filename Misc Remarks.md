- **Standardize** the features prior to training a model that uses **regularization**. 
Because in linear regression the value of the weights is partially determined by the scale of the feature, 
and in regularized models all weights are summed together.
- By default, after identifying the best hyperparameters, **`GridSearchCV`** will retrain a model using the best hyperparameters on the entire dataset (rather than leaving a fold
out for cross-validation). We can use this model to predict values like any other scikit-learn model.
- ([H2O](http://docs.h2o.ai/h2o/latest-stable/h2o-docs/data-science/gbm-faq/about_the_data.html#lossfunction)) "The GBM algorithm is quite good at handling highly **imbalanced data** because it’s simply a partitioning scheme. Use the weights column for per-row weights if you want to control over/under-sampling. When your dataset includes imbalanced data, you can also specify `balance_classes`, `class_sampling_factors`, `sample_rate_per_class`, and `max_after_balance_size` to control over/under-sampling."
- The following can be handy for data cleanup:
````
import string
string.punctuation  # '!"#$%&\'()*+,-./:;<=>?@[\\]^_`{|}~'
````
