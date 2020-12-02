**The material is for my personal use and I don't take credit for the material.*

Mind Design II (1997)  
- Chess is *digital* but billiard is not. (p 10)
- We measure the intelligence of a system by its ability to achieve stated ends in the face of variations, difficulties, and complexities posed by the task environment. (p 83)
- Neurons operate in the timescale of miliseconds, whereas computer componenets  operate in the time scale of nanoseconds--a factor of 10^6 faster. This means that human processes that take of the order of a second or less can involve only a hundred or so timesteps. ... Given that the processes we seek to characterize are often quite complex and may involve considerations of large numbers of simultanous constraints, our algorithms *must* involve considerable parallelism. (p 207)
- From conventional programmable computers we are used to thinking of knowledge  as being stored in the states of certain units in the system [e.g. RAM]. In our systems [i.e. connectionist approach] we assume that only very short-term storage can occur in the states of units; long-term storage takes place in the connections among units. ... almost all the knowledge in *implicit* in the structure of the device that carries out the task, rather than *explicit* in the states of units themselves.  
Knowledge is not directly accessible to interpretation by some separate processor, but it is built into the processor itself and directly determines the course of perocessing. It is acquired through tuning of connections, as they are used in prcessing, rather than formulated and stored as declarative facts. (p 208)
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


