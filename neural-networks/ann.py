import os
import glob
import matplotlib.pyplot as plt
import seaborn as sns

files = glob.glob('./data/confusion/*')
for f in files:
    os.remove(f)

files = glob.glob('./data/metrics/*')
for f in files:
    os.remove(f)

import numpy as np
import tensorflow as tf
keras = tf.keras
from keras.datasets import mnist
from keras.layers.core import Dense, Activation
from keras.utils import np_utils
from sklearn.metrics import confusion_matrix
import pandas as pd

num_classes = 10
input_shape = (784,1)

(x_train, y_train), (x_test, y_test) = mnist.load_data()

x_train = x_train.reshape(60000,784)
x_test = x_test.reshape(10000,784)

x_train = x_train.astype("float32")/255
x_test = x_test.astype("float32")/255

x_train = np.expand_dims(x_train,-1)
x_test = np.expand_dims(x_test,-1)

y_train = np_utils.to_categorical(y_train,num_classes)
y_test = np_utils.to_categorical(y_test,num_classes)

# Training all the different models
hidden_layers = [2, 3]
neurons = [100,150]
activation_functions = ['tanh', 'sigmoid', 'relu']

#Making model
batch_size = 128
epochs = 15
input_shape = (784,)

# Layer 1: tanh/relu/sigmoid
# Layer 2: sigmoid/sigmoid/tanh
# Layer 3: relu/tanh/relu

for activation_function in activation_functions:
    for neuron in neurons:
        for hidden_layer in hidden_layers:
            model = keras.Sequential()
            for i in range(hidden_layer-1):
                if i==0:
                    model.add(Dense(neuron/hidden_layer, input_shape=(784,)))
                    model.add(Activation(activation_function))
                else:
                    model.add(Dense(neuron/hidden_layer))
                    model.add(Activation(activation_function))
                model.add(Dense(neuron%hidden_layer + neuron/hidden_layer))
                model.add(Activation(activation_function))
            model.add(Dense(10))
            model.add(Activation('softmax'))
            model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy",keras.metrics.Recall(),keras.metrics.Precision()])
            history = model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs)
            y_pred = model.predict(x_test)
            y_pred = np.argmax(y_pred, axis=1)
            confusion = confusion_matrix(y_true=y_test.argmax(axis=1), y_pred=y_pred)
            metrics = model.evaluate(x_test, y_test)
            metrics = pd.Series(metrics,index=['Loss','Accuracy','Precision','Recall'])
            confusion = pd.DataFrame(confusion)
            print(confusion)
            confusion.to_csv("./data/confusion/"+str(activation_function)+"-"+str(hidden_layer)+"-"+str(neuron)+".csv")
            metrics.to_csv("./data/metrics/"+str(activation_function)+"-"+str(hidden_layer)+"-"+str(neuron)+".csv",header=False)


# 3 more models

hidden_layer = 3
neuron = 150
activation_function = [['tanh', 'sigmoid', 'relu'], ['relu', 'sigmoid', 'tanh'], ['sigmoid', 'tanh', 'relu']]
for p in range(3):
    model = keras.Sequential()
    for i in range(hidden_layer-1):
        if i==0:
            model.add(Dense(neuron/hidden_layer, input_shape=(784,)))
            model.add(Activation(activation_function[p][0]))
        else:
            model.add(Dense(neuron/hidden_layer))
            model.add(Activation(activation_function[p][1]))
        model.add(Dense(neuron%hidden_layer + neuron/hidden_layer))
        model.add(Activation(activation_function[p][2]))
    model.add(Dense(10))
    model.add(Activation('softmax'))
    model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy",keras.metrics.Recall(),keras.metrics.Precision()])
    history = model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs)
    y_pred = model.predict(x_test)
    y_pred = np.argmax(y_pred, axis=1)
    confusion = confusion_matrix(y_true=y_test.argmax(axis=1), y_pred=y_pred)
    metrics = model.evaluate(x_test, y_test)
    metrics = pd.Series(metrics,index=['Loss','Accuracy','Precision','Recall'])
    confusion = pd.DataFrame(confusion)
    confusion.to_csv("./data/confusion/"+str(activation_function[p])+"-"+str(hidden_layer)+"-"+str(neuron)+".csv")
    metrics.to_csv("./data/metrics/"+str(activation_function[p])+"-"+str(hidden_layer)+"-"+str(neuron)+".csv",header=False)
