from __future__ import division, print_function, absolute_import

import tflearn
from tflearn.data_utils import to_categorical, pad_sequences
#from tflearn.datasets import imdb
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.embedding_ops import embedding
from tflearn.layers.recurrent import bidirectional_rnn, BasicLSTMCell
from tflearn.layers.estimator import regression

# IMDB Dataset loading
#train, test, _ = imdb.load_data(path='imdb.pkl', n_words=10000,
#                                valid_portion=0.1)
trainX, trainY = train
testX, testY = test

# Data preprocessing
# Sequence padding
trainX = pad_sequences(trainX, maxlen=200, value=0.)
testX = pad_sequences(testX, maxlen=200, value=0.)
# Converting labels to binary vectors
trainY = to_categorical(trainY, nb_classes=2)
testY = to_categorical(testY, nb_classes=2)

# Network building
net = input_data(shape=[None, 200])	#same shape as the max length
net = embedding(net, input_dim=20000, output_dim=128)	#creates embedding matrix. Not sure if I need this for project 4
net = bidirectional_rnn(net, BasicLSTMCell(128), BasicLSTMCell(128)) #has two LSTMs with 128 units go in forward and backward directions. 
net = dropout(net, 0.5)	#dropout with keep probability of 0.5. All are finalized
net = fully_connected(net, 2, activation='softmax') #makes softmax probabilities over 2 categories, true and false
net = regression(net, optimizer='adam', loss='categorical_crossentropy')	#runs adam optimizer on net minimizing catagorical cross entropy loss

# Training
model = tflearn.DNN(net, clip_gradients=5, tensorboard_verbose=1)	# clips gradients at 5. Prints out loss, accuracy and gradients. All are finalized
model.fit(trainX, trainY, n_epoch=10, validation_set=0.05, show_metric=True, batch_size=32) #trains on train data for 10 epochs reserving 5% for dev
																		#t showing accuracy at every stap with batch size of 64.   
																		#show_metric and batch_size are finalized. Not sure about incorporating validation set
																		#or # of epochs as a time concern.