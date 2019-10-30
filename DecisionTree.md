## Decision Tree

[Source 1](https://www.nltk.org/book/ch06.html)
- A **decision stump** is a decision tree with a single node that decides how to classify inputs based on a single feature. It contains one leaf for each possible feature value, specifying the class label that should be assigned to inputs whose features have that value. We can build a decision stump based on a selected feature by assigning a label to each leaf based on the most frequent label for the selected examples in the training set (i.e., the examples where the selected feature has that value).  
- The simplest method for choosing the feature to build the decision stump, is to just build a decision stump for each possible feature, and see which one achieves the highest accuracy on the training data.  
- Given the algorithm for choosing decision stumps, the algorithm for growing larger decision trees is straightforward. We begin by one best decision stump. Leaves that do not achieve sufficient accuracy are then replaced by new decision stumps, trained on the subset of the training corpus that is selected by the path to the leaf.
- Another approach for feature selection is to select one that causes the most **information gain**. Information gain measures how much more organized the input values become when we divide them up using a given feature. It is calculated by taking the diffrence of entropy of the original set of input values' labels, before and after applying a decision stump.  The higher the information gain, the better job the decision stump does of dividing the input values into coherent groups.
- Decision trees are especially well suited to cases where many hierarchical categorical distinctions can be made. For example, decision trees can be very effective at capturing phylogeny trees.
- One problem is that, since each branch in the decision tree splits the training data, the amount of training data available to train nodes lower in the tree can become quite small. As a result, these lower decision nodes may overfit the training set. One solution to this problem is to stop dividing nodes once the amount of training data becomes too small. Another solution is to grow a full decision tree, but then to prune decision nodes that do not improve performance on a dev-test.
- Another problem is that decision trees are not good at making use of features that are weak predictors of the correct label. Since these features make relatively small incremental improvements, they tend to occur very low in the decision tree. But by the time the decision tree learner has descended far enough to use these features, there is not enough training data left to reliably determine what effect they should have. 
- The fact that decision trees require that features be checked in a specific order limits their ability to exploit features that are relatively independent of one another. The naive Bayes classification method, overcomes this limitation by allowing all features to act "in parallel."
- **Naive Bayes Classifier**: Naive Bayes begins by calculating the prior probability of each label, based on how frequently each label occurs in the training data. Every feature then contributes to the likelihood estimate for each label, by multiplying it by the probability that input values with that label will have that feature. The resulting likelihood score can be thought of as an estimate of the probability that a randomly selected value from the training set would have both the given label and the set of features, assuming that the feature probabilities are all independent.  
<a href="https://www.codecogs.com/eqnedit.php?latex=P(features,&space;label)&space;=&space;P(label)&space;\times&space;P(features|label)" target="_blank"><img src="https://latex.codecogs.com/gif.latex?P(features,&space;label)&space;=&space;P(label)&space;\times&space;P(features|label)" title="P(features, label) = P(label) \times P(features|label)" /></a>  
<a href="https://www.codecogs.com/eqnedit.php?latex=P(features,&space;label)&space;=&space;P(label)&space;\times&space;\prod\limits_{f\in&space;features}P(f|label)" target="_blank"><img src="https://latex.codecogs.com/gif.latex?P(features,&space;label)&space;=&space;P(label)&space;\times&space;\prod\limits_{f\in&space;features}P(f|label)" title="P(features, label) = P(label) \times \prod\limits_{f\in features}P(f|label)" /></a>

[Source 2](https://scikit-learn.org/stable/modules/tree.html)
- Requires little data preparation (normalisation not needed)
- Able to handle both numerical and categorical data (sklearn version doesn't support categorical yet).
- Pruning (not currently supported), setting the minimum number of samples required at a leaf node or setting the maximum depth of the tree are can be used to prevent **overfitting**.
- Decision trees can be unstable because small variations in the data might result in a completely different tree being generated. This problem is mitigated by using decision trees within an **ensemble**.
- There are concepts that are hard to learn because decision trees do not express them easily, such as XOR, parity or multiplexer problems.
- Decision tree learners create biased trees if some classes dominate. It is therefore recommended to **balance the dataset** prior to fitting with the decision tree.
- Consider performing dimensionality reduction (**PCA**, **ICA**, or **Feature selection**) beforehand to give your tree a better chance of finding features that are discriminative.
- Decision trees tend to overfit on data with a large number of features. Getting the right ratio of samples to number of features is important. Use `max_depth` to control the size of the tree to prevent overfitting.
- Use `min_samples_split` or `min_samples_leaf` to ensure that multiple samples inform every decision in the tree, by controlling which splits will be considered. A very small number will usually mean the tree will overfit, whereas a large number will prevent the tree from learning the data. Try `min_samples_leaf=5` as an initial value. If the sample size varies greatly, a float number can be used as percentage in these two parameters. While `min_samples_split` can create arbitrarily small leaves, `min_samples_leaf` guarantees that each leaf has a minimum size, avoiding low-variance, over-fit leaf nodes in regression problems. For classification with few classes, `min_samples_leaf=1` is often the best choice.
- If the input matrix X is very sparse, it is recommended to convert to sparse `csc_matrix` before calling fit and sparse `csr_matrix` before calling predict. Training time can be orders of magnitude faster.

Visualizing a decision tree
````Python
from sklearn.datasets import load_iris
from sklearn import tree
import matplotlib.pyplot as plt

iris = load_iris()
clf = tree.DecisionTreeClassifier()  # For Regression use DecisionTreeRegressor()
clf.fit(iris.data, iris.target)
clf.predict([[5, 5, 5, 5]])
# To get the probabilities of each class
clf.predict_proba([[5, 5, 5, 5]])

plt.figure(figsize=(20,15))
tree.plot_tree(clf \
               , feature_names=iris.feature_names \
               , class_names=iris.target_names \
               , filled=True \
               , rounded=True \
               , fontsize=14) 
              # 'proportion' option is also usefull; 
              # changes the display of ‘values’ and/or ‘samples’ to be proportions and percentages respectively.
plt.show()
````
Feature Importance
````Python
for name, importance in zip(iris.feature_names, clf.feature_importances_):
  print(name, importance)
````
To check the data
````Python
dir(iris)  # >>> ['DESCR', 'data', 'feature_names', 'filename', 'target', 'target_names']
print(*iris.feature_names, 'Label', sep = ',')  # Header (column names)
for features_values, label_code in zip(iris.data, iris.target):
  print(features_values, iris.target_names[label_code])
````
[Multi-output problems](https://scikit-learn.org/stable/modules/tree.html#multi-output-problems)

### Splitting
- An ordered variable *X* with *n* unique values yields `n−1` splits of the form `X ≤ c`
- A categorical variable with *m* unique values yields `(2^(m−1))−1` splits of the form `X ∈ A`
