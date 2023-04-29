import numpy as np
import tensorflow as tf
keras = tf.keras
from keras.datasets import mnist
from keras.models import Sequential, load_model
from keras.layers.core import Dense, Dropout, Activation
from keras.utils import np_utils

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

for activation_function in activation_functions:
    for neuron in neurons:
        for hidden_layer in hidden_layers: 
            model = keras.Sequential()
            # print("model details:")
            # print(activation_function)
            # print(neuron)
            # print(hidden_layer)
            for no in range(1, hidden_layer):
                model.add(Dense(neuron/hidden_layer,input_shape=(784,)))
                model.add(Activation(activation_function))
            
            model.add(Dense(neuron/hidden_layer,input_shape=(784,)))
            model.add(Activation(activation_function))
                

            model.add(Dense(10))
            model.add(Activation('softmax'))

            model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy",keras.metrics.Recall(),keras.metrics.Precision()])

            history = model.fit(
                x_train, y_train, batch_size=batch_size, epochs=epochs
            )

            metrics = model.evaluate(x_test,y_test,verbose=2)

            print(metrics[0])
            print(metrics[1])
            print(metrics[2])
            print(metrics[3])