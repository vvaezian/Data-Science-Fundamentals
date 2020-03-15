**The material is for my personal use and I don't take credit for the material.*

Mind Design II (1997)
- Chess is *digital* but billiard is not. (p 10)
- We measure the intelligence of a system by its ability to achieve stated ends in the face of variations, difficulties, and complexities posed by the task environment. (p 83)

#### Workflow
1- Work on reducing the training error, until it is in an acceptable range.
  - If training error is high (compared to the acceptable error (e.g. human error)) then the model has high bias.  
We should try bigger model, train longer or try new model architecture.

2- Work on dev (aka validation) error, until it is in an acceptable range.
  - If dev error is high, then the model has high variance.  
We should try more data, regularization or try new model architecture.

# Remarks
- Dev and test sets must be from the same distribution. So if we 50000 hours train data and 10 hours test data, the dev set must be 5 hours of the test data.
