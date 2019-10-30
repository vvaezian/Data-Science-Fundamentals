- A regular convolutional filter window is a box of cells that slides through data and dot product of window cells is calculated and the result is written in the next layer in place of the window.
- Down-Sampling
  - **Max Pooling**: A filter that replaces a frame with max value of the frame's cells. It then moves 'strides' cells to the right (or down to beginning of next row if reached end of row), and repeat.
  - **Average Pooling**
- Up-Sampling
  - **Nearest Neighbor Unpooling**: A 2x2 window W1 can become a 4x4 window W2 where top-left 2x2 of W2 is all taken from top-left cell of W1, and so on
  - **Bed of Nails Unpooling**: The extra cells are zero, and the cell with value is chosen randomely (?)
  - **MAX Unpooling**: Similar to Bed of Nails but the cell with value is chosen from the Max Pooling level (so when max pooling is done the model remembers which cell was chosen, and uses it for max unpooling step). 
````python
import keras
from keras.models import Sequential
from keras.layers import Activation
from keras.layers.core import Dense, Flatten
from keras.layers.convolutional import *
from keras.layers.pooling import *

model = Sequential([
  Dense(16, activation='relu', input_shape=(20, 20, 3)),
  Conv2D(32, kernel_size=(3, 3), activation='relu', padding='same'),
  # kernel_size is the width and height of filter window 
  # In padding, 'valid' means not adding zero padding 
  # and 'same' means keeping the window size by ading zero padding (before applying the filter)
  MaxPooling2D(pool_size=(2, 2), strides=2, padding='valid'),
  Conv2D(64, kernel_size=(5, 5), activation='relu', padding='same'),
  Flatten(),
  Dense(2, activation='softmax')
])

model.summary()
model.get_weights()
````
