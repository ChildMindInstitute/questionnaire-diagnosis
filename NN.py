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

mlb = preprocessing.MultiLabelBinarizer()
# bin = mlb.fit_transform(['Somatic Symptom and Related Disorders', 'Attention-Deficit/Hyperactivity Disorder', 'Specific Learning Disorder', 'Intellectual Disability', 'Substance Related and Addictive Disorders', 'Elimination Disorders', 'Sleep-Wake Disorders', 'Autism Spectrum Disorder', 'Disruptive, Impulse Control and Conduct Disorders', 'Anxiety Disorders', 'Schizophrenia Spectrum and other Psychotic Disorders', 'Motor Disorder', 'Depressive Disorders', 'Feeding and Eating Disorders', 'No Diagnosis Given', 'Gender Dysphoria', 'Bipolar and Related Disorders', 'Trauma and Stressor Related Disorders', 'Personality Disorders', 'Obsessive Compulsive and Related Disorders', 'Other Conditions That May Be a Focus of Clinical Attention', 'Communication Disorder'])
bin = mlb.fit_transform(df['Dx'])

print(np.array(df['Dx']))

sample = np.array(df['Dx'])
sample = mlb.fit_transform(sample)
print(sample[0])

df['labelled'] = list(bin)

num_col = df.shape[1] - 4
num_row = df.shape[0]
Dx_types = ['Number of different types of Dx']

train_set = df[df['train'] == True]
test_set = df[df['train'] == False]

tr_array = train_set.values
te_array = test_set.values

row = ['Index of column that contains last feature']

tr_inputs = tr_array[:, 1:-3]
te_inputs = te_array[:, 1:-3]
tr_outputs = tr_array[:, -1:]
te_outputs = te_array[:, -1:]

tr_row = tr_array.shape[0]
print(tr_inputs.shape)
print(tr_outputs.shape)


def HBN_model():
    model = Sequential()
    model.add(Dense(tr_row, input_shape=(num_col,), activation='relu'))
    model.add(Dropout(.1))
    model.add(Dense(len(bin), activation='sigmoid'))
    # model.compile(loss='binary_crossentropy', optimizer=sgd)
    model.compile(loss='binary_crossentropy', optimizer='adam')
    return model

model = HBN_model()
model.fit(tr_inputs, tr_outputs, epochs=10, batch_size=50)
preds = model.predict(te_inputs)
preds[preds >= .5] = 1
preds[preds < 0.5] = 0

# Compare output of predictions in 'preds' and targets in 'test_target'