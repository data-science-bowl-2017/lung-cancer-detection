
# coding: utf-8

# In[1]:

__author__ = 'Minhaz Palasara'

import tensorflow
import keras
import shapes_3d
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.convolutional import Convolution3D, MaxPooling3D
from keras.optimizers import SGD, RMSprop
from keras.utils import np_utils, generic_utils


# In[2]:

"""
To classify/track 3D shapes, such as human hands (http://www.dbs.ifi.lmu.de/~yu_k/icml2010_3dcnn.pdf),
we first need to find a distinct set of features. Specifically for 3D shapes, robust classification can be done using
3D features.
Features can be extracted by applying a 3D filters. We can auto learn these filters using 3D deep learning.
This example trains a simple network for classifying 3D shapes (Spheres, and Cubes).
GPU run command:
    THEANO_FLAGS=mode=FAST_RUN,device=gpu,floatX=float32 python shapes_3d_cnn.py
CPU run command:
    THEANO_FLAGS=mode=FAST_RUN,device=cpu,floatX=float32 python shapes_3d_cnn.py
For 4000 training samples and 1000 test samples.
90% accuracy reached after 40 epochs, 37 seconds/epoch on GTX Titan
"""

# Data Generation parameters
test_split = 0.2
dataset_size = 5000
patch_size = 32

(X_train, Y_train),(X_test, Y_test) = shapes_3d.load_data(test_split=test_split,
                                                          dataset_size=dataset_size,
                                                          patch_size=patch_size)

# CNN Training parameters
batch_size = 10
nb_classes = 2
nb_epoch = 100

# convert class vectors to binary class matrices
Y_train = np_utils.to_categorical(Y_train, nb_classes)
Y_test = np_utils.to_categorical(Y_test, nb_classes)

# number of convolutional filters to use at each layer
nb_filters = [16, 32]

# level of pooling to perform at each layer (POOL x POOL)
nb_pool = [3, 3]

# level of convolution to perform at each layer (CONV x CONV)
nb_conv = [7, 3]

model = Sequential()

model.add(Convolution3D(
    nb_filters[0],
    nb_conv[0],
    nb_conv[0],
    nb_conv[0],
    border_mode='same',
    input_shape=(1, patch_size, patch_size, patch_size),
    activation='relu')
)

model.add(MaxPooling3D(pool_size=(1, nb_pool[0], nb_pool[0])))

model.add(Dropout(0.5))
model.add(Convolution3D(
    nb_filters[1],
    nb_conv[1],
    nb_conv[1],
    nb_conv[1], border_mode='same',
    activation='relu')
)

model.add(MaxPooling3D(pool_size=(1, nb_pool[1], nb_pool[1])))
model.add(Flatten())
model.add(Dropout(0.5))
model.add(Dense(16, init='normal', activation='relu'))

model.add(Dense(nb_classes, init='normal'))
model.add(Activation('softmax'))

sgd = RMSprop(lr=0.001, rho=0.9, epsilon=1e-08, decay=0.0)
model.compile(loss='categorical_crossentropy', optimizer=sgd)

model.fit(
    X_train,
    Y_train,
    batch_size=batch_size,
    nb_epoch=nb_epoch,
    metrics=["accuracy"],
    verbose=2,
    validation_data=(X_test, Y_test)
)

score = model.evaluate(X_test, Y_test, batch_size=batch_size, metrics=["accuracy"])

print('Test score:', score[0])
print('Test accuracy:', score[1])


# In[3]:

model.summary()


# In[ ]:



