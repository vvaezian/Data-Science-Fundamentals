**The material is for my personal use and I don't take credit for the material.*

Mind Design II (1997)  
- Chess is *digital* but billiard is not. (p 10)
- We measure the intelligence of a system by its ability to achieve stated ends in the face of variations, difficulties, and complexities posed by the task environment. (p 83)
- Neurons operate in the timescale of miliseconds, whereas computer componenets  operate in the time scale of nanoseconds--a factor of 10^6 faster. This means that human processes that take of the order of a second or less can involve only a hundred or so timesteps (p 207).

-----------------

Andrew Ng's talk  
#### Workflow
1. Work on reducing the training error, until it is in an acceptable range.
  - If training error is high (compared to the acceptable error (e.g. human error)) then the model has high bias.  
We should try bigger model, train longer or try new model architecture.

2. Work on dev (aka validation) error, until it is in an acceptable range.
  - If dev error is high, then the model has high variance.  
We should try more data, regularization or try new model architecture.

#### More granular Workflow
Dev and test sets must be from the same distribution. So if we have 50000 hours train data and 10 hours test data, then dev set must be 5 hours of the test data.  
It also helps to consider a portion of train set as train-dev set (20 hour in the above example). So we have train/train-dev/dev/test sets. The workflow is as follows:

1. train error. same as above
2. train-dev error. same as dev error above
3. dev error. More data, data synthesis, new model.
4. test error. Get more dev data.


