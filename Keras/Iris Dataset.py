
"""Modelling the Iris dataset with neural nets using keras library."""

import numpy as np
import keras
from keras import backend as K
from keras.models import Sequential
from keras.layers import Activation
from keras.layers.core import Dense
from keras.optimizers import Adam, SGD
from keras.metrics import categorical_crossentropy
# from keras import regularizers
import pandas as pd
from sklearn.datasets import load_iris

# mdata = np.loadtxt('C:\\Users\Vahid\Dropbox\Python Projects\digitizedData', delimiter=',')
mdata = load_iris()
mx = mdata.data 
my = [[i] for i in  mdata.target] # to match the shape of mx for concatenation in the next step
data = np.concatenate((mx, my), axis=1) # concatenate to be able to shuffle
np.random.shuffle(data) # this is required for validation_split. Because it only get data from the end portion of data
x = data[:, :-1] # training and validation data (first four columns)
y = data[:, -1] # labels (fifth column)
#print(x.shape)

# df = pd.DataFrame.from_records(data=mdata.data, columns=mdata.feature_names)
# df['species'] = mdata.target
# pd = shuffle_dataframe(pd)

# from keras.utils import to_categorical
# y = to_categorical(y, num_classes=3)

model = Sequential([
  Dense(16, input_shape=(4,), activation='relu'), # first hidden layer
  Dense(32, activation='relu'), # second hidden layer
  Dense(3, activation='softmax') # output layer. The numbers 3 is for 3 classes of flowers that we have
  # We can add regularizer to some layers: e.x. 'kernel_regularizer=regularizers.l2(0.01)
  # Regularizers penalize large weights in the model. It helps to avoid overfitting
])
# # We can also add 'dropouts' to avoid overfitting. They randomly ignore some neurons temporarily.
# from keras.layers import Dropout
# model.add(Dropout(0.2))

# sgd = SGD(lr=0.1, momentum=0.9, decay=0.0001, nesterov=False)
model.compile(optimizer=Adam(lr=.01), loss='sparse_categorical_crossentropy', metrics=['accuracy'])
print(model.summary())
# 'Adam' is a variation of sgd (Stochastic Gradient Descent)
# its lr (learning rate) can be set individually by: model.optimizer.lr = 0.01

model.fit(x, y, epochs=10, batch_size=10, shuffle=False, verbose=1, validation_split=0.2)
# In gradient descent algorithms, the sum of gradients with respect to several examples is calculated
# and then using this cumulative gradient, the parameters (weights and biases) are updated. 
# If it uses all training examples before one ‘update’, then it’s called full batch learning. 
# If it uses only one example, it’s called stochastic gradient descent (online learning). 
# If it uses a small batch of examples, it’s often called mini-batch learning.
# batch_size is the number of input instances to be fed to the sgd before one update.

test_data = np.array([[5,3,1,0.1]])
predictions = model.predict(test_data, verbose=0)
# 'predict' outputs probabilities of each class (I guess if the last layer has softmax activation)
# 'predict_classes' outputs just one class
 
for i in predictions:
  print(i)
