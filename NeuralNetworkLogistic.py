# -*- coding: utf-8 -*-
"""
Created on Sat Feb 23 16:41:15 2019

@author: Sahil Nathani
"""

import numpy as np
import pandas as pd
#from sklearn.preprocessing import OneHotEncoder
#from sklearn.metrics import accuracy_score

df1 = pd.read_csv('C:\\Users\\Sahil Nathani\\Desktop\\Python and ML Material\\Databases\\Coursera\\courseraLogistic1.csv')
df2 = pd.read_csv('C:\\Users\\Sahil Nathani\\Desktop\\Python and ML Material\\Databases\\Coursera\\courseraLogistic2.csv')
'''
def sigmoid(x):
    return 1/(1+np.exp(-x))

def sigmoid_deri(x):
    return sigmoid(x)*(1-sigmoid(x))

class NeuralNetwork:
    
    def __init__(self, x, y):
        self.input = x
        self.w1 = np.random.rand(self.input.shape[1], 4)
        self.w2 = np.random.rand(4, 1)
        self.y = y
        self.output = np.random.rand(y.shape[0], y.shape[1])
        
    def feedforward(self):
        self.layer1 = sigmoid(np.dot(self.input, self.w1))
        #self.layer2 = sigmoid(np.dot(self.layer1, self.w2))
        
    def backpropagate(self):
        d_w2 = np.dot(self.layer1.T, (2*(self.y-self.output)*sigmoid_deri(self.output)))
        d_w1 = np.dot(self.input.T, (np.dot(2*(self.y-self.output)*sigmoid_deri(self.output), self.w2.T)*sigmoid_deri(self.layer1)))
        self.w1+=d_w1
        self.w2+=d_w2

X = np.array(df1.drop(['Decision'], 1))       
Y = np.array(df1['Decision']).reshape(100, 1)

nn = NeuralNetwork(X, Y)
nn.feedforward()
nn.backpropagate() 
''' 

#tensorflow approach
import tensorflow as tf
from sklearn import cross_validation
 
X = np.array(df1.drop(['Decision'], 1))       
Y = np.array(df1['Decision'])

X_train, X_test, Y_train, Y_test = cross_validation.train_test_split(X, Y, test_size=1)

X_train = tf.keras.utils.normalize(X_train, axis=1)
X_test = tf.keras.utils.normalize(X_test, axis=1)

model = tf.keras.Sequential()

#model.add(tf.keras.layers.Flatten())
model.add(tf.keras.layers.Dense(256, input_dim=2,activation=tf.nn.relu))
model.add(tf.keras.layers.BatchNormalization())
model.add(tf.keras.layers.Dense(256, activation=tf.nn.relu))
model.add(tf.keras.layers.BatchNormalization())
#model.add(tf.keras.layers.Dropout(0.2))
model.add(tf.keras.layers.Dense(1, activation=tf.nn.softmax))

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

model.fit(X_train, Y_train, epochs=10)
val_loss, val_acc= model.evaluate(X_test, Y_test)

X_new = np.array([[79, 75], [35, 72]])

print(model.predict_classes(X_new))