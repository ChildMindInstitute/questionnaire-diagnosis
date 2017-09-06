# This script is intended to implement neural networks to the HBN dataset

# from keras.datasets import mnist
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.utils import np_utils
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import MaxPooling2D
import pandas as pd
from keras import backend as K
from keras.optimizers import SGD
K.set_image_dim_ordering('th')
from sklearn import preprocessing
import keras
from sklearn.preprocessing import MultiLabelBinarizer
from ast import literal_eval

########################################################################################################################

# Create a simple neural network with one hidden layer

# (X_train, y_train), (X_test, y_test) = mnist.load_data()
#
# seed = 7
# np.random.seed(seed)
#
# num_pixels = X_train.shape[1] * X_train.shape[2]
# X_train = X_train.reshape(X_train.shape[0], num_pixels).astype('float32')
# X_test = X_test.reshape(X_test.shape[0], num_pixels).astype('float32')
#
# X_train = X_train / 255
# X_test = X_test / 255
#
# y_train = np_utils.to_categorical(y_train)
# y_test = np_utils.to_categorical(y_test)
# num_classes = y_test.shape[1]

# def baseline_model():
#     model = Sequential()
#     model.add(Dense(num_pixels, input_dim=num_pixels, kernel_initializer='normal', activation='relu'))
#     model.add(Dense(num_classes, kernel_initializer='normal', activation='softmax'))
#     model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
#     return model
#
# model = baseline_model()
# model.fit(X_train, y_train, validation_data = (X_test, y_test), epochs=10, batch_size=200, verbose=2)
# scores = model.evaluate(X_test, y_test, verbose=0)
# print('Baseline Error: %.2f%%' % (100-scores[1]*100))

########################################################################################################################

# Create a Convolutional Neural Network

# seed = 7
# np.random.seed(seed)
# # load data
# (X_train, y_train), (X_test, y_test) = mnist.load_data()
# # reshape to be [samples][pixels][width][height]
# X_train = X_train.reshape(X_train.shape[0], 1, 28, 28).astype('float32')
# X_test = X_test.reshape(X_test.shape[0], 1, 28, 28).astype('float32')
# # normalize inputs from 0-255 to 0-1
# X_train = X_train / 255
# X_test = X_test / 255
# # one hot encode outputs
# y_train = np_utils.to_categorical(y_train)
# y_test = np_utils.to_categorical(y_test)
# num_classes = y_test.shape[1]
#
# def baseline_model():
#     model = Sequential()
#     # Create the initial layer with 32 neurons
#     model.add(Conv2D(32, (5,5), input_shape=(1, 28, 28), activation='relu'))
#     # Pooling clusters outputs of multiple neurons in one layer to one neuron in the next layer
#     model.add(MaxPooling2D(pool_size=(2, 2)))
#     # Randomly exclude 20% of neurons in the layer to reduce overfitting
#     # Referred to as a regularization layer
#     model.add(Dropout(0.2))
#     model.add(Flatten())
#     # Add another layer with 128 neurons
#     model.add(Dense(128, activation='relu'))
#     # softmax allows us to analyze probabilities of each prediction
#     # Add the output layer with one neuron for each potential digit represented
#     model.add(Dense(num_classes, activation='softmax'))
#     model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
#     return model
#
# model = baseline_model()
# model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=10, batch_size=200, verbose=2)
#
# scores = model.evaluate(X_test, y_test, verbose=0)
# print('Baseline Error: %.2f%%' % (100-scores[1]*100))

########################################################################################################################

# Complex Convolutional Neural Network

# seed = 7
# np.random.seed(seed)
# # load data
# (X_train, y_train), (X_test, y_test) = mnist.load_data()
# # reshape to be [samples][pixels][width][height]
# X_train = X_train.reshape(X_train.shape[0], 1, 28, 28).astype('float32')
# X_test = X_test.reshape(X_test.shape[0], 1, 28, 28).astype('float32')
# # normalize inputs from 0-255 to 0-1
# X_train = X_train / 255
# X_test = X_test / 255
# # one hot encode outputs
# y_train = np_utils.to_categorical(y_train)
# y_test = np_utils.to_categorical(y_test)
# num_classes = y_test.shape[1]
#
# def larger_model():
#     # create model
#     model = Sequential()
#     model.add(Conv2D(30, (5, 5), input_shape=(1, 28, 28), activation='relu'))
#     model.add(MaxPooling2D(pool_size=(2, 2)))
#     model.add(Conv2D(15, (3, 3), activation='relu'))
#     model.add(MaxPooling2D(pool_size=(2, 2)))
#     model.add(Dropout(0.2))
#     model.add(Flatten())
#     model.add(Dense(128, activation='relu'))
#     model.add(Dense(50, activation='relu'))
#     model.add(Dense(num_classes, activation='softmax'))
#     # Compile model
#     model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
#     return model
#
# model = larger_model()
# model.fit(X_train, y_train, epochs=10, batch_size=200)
# scores = model.evaluate(X_test, y_test, verbose=1)
# print("Large CNN Error: %.2f%%" % (100-scores[1]*100))

########################################################################################################################

# Build simple neural network for HBN dataset with one hidden layer

sgd = SGD(lr=0.01, decay=1e-6, momentum=.9, nesterov=True)

filename = 'DSM_NaN_Replaced.xlsx'
sheetname = 'DSM_Data'
output = pd.ExcelFile(filename)
df = output.parse(sheetname)
df = pd.DataFrame(data=df)

train_set = df[df['train'] == True][0:100]
test_set = df[df['train'] == False][0:50]

train_targets = list(train_set['Dx'])
test_targets = list(test_set['Dx'])

X_train = train_set.columns[2:-2]
X_test = (test_set[X_train])
X_train = (train_set[X_train])

print(X_train)

for row in range(train_set.shape[0]):
    train_targets[row] = literal_eval(train_targets[row])

for row in range(test_set.shape[0]):
    test_targets[row] = literal_eval(test_targets[row])

mlb = MultiLabelBinarizer()
Y = mlb.fit_transform(train_targets)

def HBN_model():
    model = Sequential()
    model.add(Dense(150, input_shape=(606,), activation='relu'))
    model.add(Dropout(.1))
    model.add(Dense(75, activation='sigmoid'))
    model.add(Dense(24))
    # model.compile(loss='binary_crossentropy', optimizer=sgd)
    model.compile(loss='binary_crossentropy', optimizer='adam')
    return model

model = HBN_model()
print(X_train)
model.fit(np.array(X_train), Y, epochs=5, batch_size=5, verbose=1)

Z = mlb.fit_transform(test_targets)
preds = model.predict(np.array(X_test))
preds[preds >= .5] = 1
preds[preds < 0.5] = 0
print(preds[0])
print(preds[1])
print(preds[2])
print(preds[3])
print(preds[4])

# Compare output of predictions in 'preds' and targets in 'test_target'
