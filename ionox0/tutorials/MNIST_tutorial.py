
# coding: utf-8

# In[ ]:

from keras.datasets import mnist
from keras.layers.core import Dense, Dropout, Activation
from keras.optimizers import SGD, Adam, RMSprop
from keras.utils import np_utils

(X_train, y_train), (X_test, y_test) = mnist.load_data()

X_train = X_train.reshape(60000, 784)
X_test = X_test.reshape(10000, 784)
X_train = X_train.astype('float32')
X_test = X_test.astype('float32')
X_train /= 255
X_test /= 255

nb_classes = 10
Y_train = np_utils.to_categorical(y_train, nb_classes)
Y_test = np_utils.to_categorical(y_test, nb_classes)


model2 = Sequential()
model2.add(Dense(128, input_shape=(784,)))
model2.add(Activation('sigmoid'))
model2.add(Dense(10))
model2.add(Activation('softmax'))

sgd = SGD()
get_ipython().magic(u"time model2.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['mse', 'accuracy'])")
h = model2.fit(X_train, Y_train, batch_size = 128, nb_epoch=3,
               show_accuracy=True, validation_data=(X_test, Y_test),
              verbose=2)


# In[ ]:

h.history


# In[ ]:

model2.summary()


# In[ ]:

model2.weights[3].get_shape()


# In[ ]:

W1, b1, W2, b2 = model2.get_weights()


# In[ ]:

b2.shape

