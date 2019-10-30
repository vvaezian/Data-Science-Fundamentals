### Saving the trained model with all the weights etc
````Python
model.save('name.h5')
new_model = load_model('name.h5')
````
**Checkpointing** can be usefulto keep track of the model weights in case your training run is stopped prematurely.
It is alsouseful to keep track of the best model observed during training.

````Python
from keras.callbacks import ModelCheckpoint
checkpoint = ModelCheckpoint('weights.best.hdf5', monitor='val_acc', save_best_only=True, mode='max')
callbacks_list = [checkpoint]
model.fit(..., callbacks=callbacks_list)
````
### Saving only the architecture, not the weights or training config
````Python
jsonString = model.to_json()
yamString = model.to_yaml()

new_model_architecture = model_from_json(jsonString)
````
### Saving only the weights
````Python
model.save_weights('name.h5')

new_model = Sequentioal(...)
new_model.load_weights('name.h5')
````
### To get reproducible results
````Python
import numpy as np
import tensorflow as tf
import random as rn
import os

os.environ['PYTHONHASHSEED'] = '0'
np.random.seed(1)
rn.seed(2)
tf.set_random_seed(3)
from keras import backend as K
session_conf = tf.ConfigProto(intra_op_parallelism_threads=1, inter_op_parallelism_threads=1)
sess = tf.Session(graph=tf.get_default_graph(), config=session_conf)
K.set_session(sess)
````
### To use the model with Scikit-learn
````Python
# Function to create model, required for KerasClassifier
def create_model():
  model = Sequential()
  ...
  model.compile(...)
return model

# create classifier for use in scikit-learn
model = KerasClassifier(build_fn=create_model, epochs=150, batch_size=10)
# evaluate model using 10-fold cross-validation in scikit-learn
kfold = StratifiedKFold(n_splits=10, shuffle=True, random_state=seed)
results = cross_val_score(model, X, Y, cv=kfold)
````
