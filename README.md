**The material is for my personal use and I don't take credit for the material.*

ML interview Questions

- Cross val
- Shallow vs deep copy
- Clustering (k means?)
- Graph (find overfitting)
- Deep vs shallow layers (what each detect)
- Pretrained model
- Regularization
- Is 98% accuracy good in (10000 a 500 b)?
- Cnn difference with others
- False positive and false negative
- Accuracy, precision, recall

-------------------------------------

Mind Design II (1997)  
- Chess is *digital* but billiard is not. (p 10)
- We measure the intelligence of a system by its ability to achieve stated ends in the face of variations, difficulties, and complexities posed by the task environment. (p 83)
- Neurons operate in the timescale of miliseconds, whereas computer componenets  operate in the time scale of nanoseconds--a factor of 10^6 faster. This means that human processes that take on the order of a second or less can involve only a hundred or so timesteps. ... Given that the processes we seek to characterize are often quite complex and may involve considerations of large numbers of simultanous constraints, our algorithms *must* involve considerable parallelism. (p 207)
- From conventional programmable computers we are used to thinking of knowledge as being stored in the states of certain units in the system [e.g. RAM or Hard Drive]. In our systems [i.e. connectionist approach] we assume that only very short-term storage can occur in the states of units; long-term storage takes place in the connections among units. ... almost all the knowledge is *implicit* in the structure of the device that carries out the task, rather than *explicit* in the states of units themselves.  
Knowledge is not directly accessible to interpretation by some separate processor, but it is built into the processor itself and directly determines the course of perocessing. It is acquired through tuning of connections, as they are used in prcessing, rather than formulated and stored as declarative facts. (p 208)
- The number of connections going out of a neuron or coming into a neuron can range up to 100,000 in some parts of the brain.
- {re-read section 1.2 p237 and 1.3 about subsymbolic paradigm}  
In the symbolic approach, symbols(atoms) are used to denote the semantically interpretable entities (concepts); those same symbols are the objects governed by symbol manipulation in the rules that define the system. [In other Words] The entities which are semantically interpretable are *also* the entities governed by the formal laws that define the system. In the subsymbolic paradigm, this is no longer true. The semantically interpretable entities are *patterns of activation* over large number of units in the system, whereas the entities manipulated by formal rules are the individual activations of cells in the network. (p. 239)
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


