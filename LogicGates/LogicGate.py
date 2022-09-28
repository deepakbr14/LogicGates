#################################################################################
#
# this code is a rewrite of this stuff:
# https://broutonlab.com/blog/tutorial-create-simple-neural-network
# https://blog.thoughtram.io/machine-learning/2016/11/02/understanding-XOR-with-keras-and-tensorlow.html
#
# tf.keras doc is here:
# https://www.tensorflow.org/api_docs/python/tf/keras
# i'm running 2.6
#   https://www.tensorflow.org/versions/r2.6/api_docs/python/tf
#
#################################################################################

#dsd go through all calls

import os
import sys
import numpy as np                          # for management of arrays

from tensorflow.python import keras

from keras.models import Sequential         # keras is a neural network library that runs on top of tensorflow
                                            #   note:   keras has two different APIs to construct a model
                                            #           functional and sequential
from keras.layers.core import Dense         # Dense layer is a simple layer of neurons - each neuron gets inputs from all neurons of prev layer
                                            #   note:   bunch of different layer types are available
from si.SiLog import * 


print("running keras version [", keras.__version__, "]")

data_input  = None
data_output = None
model       = None
layer_dense = None


INPUT_LAYER_NUM_NEURONS     = 2
HIDDEN_LAYER_NUM_NEURONS    = 5
OUTPUT_LAYER_NUM_NEURONS    = 1
NUM_EPOCHS                  = 1000
VERBOSE                     = 2             # range is 0-2

# creating neural network to learn the AND truth table
#   
#   input               output
#   0   0               0
#   0   1               0
#   1   0               0
#   1   1               1
#
#   only 1 AND 1 is true
#

#create the input array
#dsd what is the float 32 for
data_input   = np.array([[0,0], [0,1], [1,0], [1,1]])


#create output array
#dsd what is the float 32 for
data_output  = np.array([[0],   [1],   [1],   [0]   ])

#dsd
#SiLog.KeyVal("training data", data_input.ndim);
#os._exit(1)


#create architecture of neural network
# will look something like this: https://broutonlab.com/ghost/content/images/blog/tutorial-create-simple-neural-network/network-design.jpg
# INPUT_LAYER_NUM_NEURONS   input neurons  (layer 0) for the 2 inputs
# HIDDEN_LAYER_NUM_NEURONS  hidden neurons (layer 1) for the learning
# OUTPUT_LAYER_NUM_NEURONS  output neuron  (layer 2) for the answer

# layer         0       1                               2
# num neurons   2       HIDDEN_LAYER_NUM_NEURONS        1

#create empty model of type sequential - which means layers will come after one another
model = Sequential()

#creates input and 1st hidden layer
#dsd not sure why this is not done separately
#activation function is relu  dsd
#note: if you don't specifiy an activation - none will be used
#      with activation : output = activation(input x weight)
#      no activation   : input x weight
layer_dense = Dense(HIDDEN_LAYER_NUM_NEURONS, input_dim=INPUT_LAYER_NUM_NEURONS, activation='relu')
model.add(layer_dense)

#creates output layer
#activation function is sigmoid dsd
layer_dense = Dense(OUTPUT_LAYER_NUM_NEURONS, activation='sigmoid')
model.add(layer_dense)


#setup training
#loss = the output difference from our goal
#       the lower the loss the better
#       0 loss means perfectly solved
#       note: there are different loss functions - we hapen to choose mean_squared_error here
#             (another possible one is binary_crossentropy)
#optimizer = finds the adjustments for the weights
#            note: also a num of diff possibilities here
#setup metrics = what metrics to collect during training
model.compile(loss='mean_squared_error', 
              optimizer='adam', 
              metrics=['binary_accuracy'])


#show the model
model.summary()

#train
model.fit(data_input, data_output, epochs=NUM_EPOCHS, verbose=VERBOSE)

#evaluation 
scores = model.evaluate(data_input, data_output)
print("\n%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))

#prediction
SiLog.KeyVal("Prediction Results Original", model.predict(data_input));
SiLog.KeyVal("Prediction Results Rounded", model.predict(data_input).round());
