# -*- coding: utf-8 -*-
"""
Created on Fri Apr 27 22:26:07 2018

@author: Myron
"""

import pandas as pd
import datetime
import numpy as np
import matplotlib.pyplot as plt
import requests
import re
import csv
import codecs
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf


#Get data
def get_google_finance_intraday(ticker):
    page = requests.get("https://www.google.com/finance/getprices?q=BABA&i=1800&p=3Y&f=d,o,h,l,c,v")
    reader = csv.reader(codecs.iterdecode(page.content.splitlines(), "utf-8"))
    columns = ['Open', 'High', 'Low', 'Close', 'Volume']
    rows = []
    times = []
    for row in reader:
        if re.match('^[a\d]', row[0]):
            if row[0].startswith('a'):
                start = datetime.datetime.fromtimestamp(int(row[0][1:]))
                times.append(start)
            else:
                times.append(start+datetime.timedelta(seconds=1800*int(row[0])))
            rows.append(map(float, row[1:]))
    if len(rows):
        return pd.DataFrame(rows,columns=columns)
#        return pd.DataFrame(rows, index=pd.DatetimeIndex(times, name='Date'),columns=columns)
    else:
#        return pd.DataFrame(rows, index=pd.DatetimeIndex(times, name='Date'))
        return pd.DataFrame(rows,columns=columns)

df = get_google_finance_intraday('BABA')


#Plot 3 years price 
#df['Time']= df.index
#df.Date= pd.to_datetime(df.Date, format='%Y-%m-%d %H:%M:%S')
#x=df['Date']
#y=df['Close']
#x=df['Time']
#y=df['Close']
#plt.plot(x,y)
#plt.grid(True)
#plt.show()

#Time and Close df

df_cut = pd.DataFrame(df['Close'])
#df_cut['Time'] = df['Time']
x=df_cut.index
y=df_cut['Close']

plt.plot(x,y)
plt.grid(True)

#Training & testing data set
n=df_cut.shape[0]
p = df_cut.shape[1]

#df = df.drop(['Date'], 1)
#df_cut =df_cut.values
df =df.values
train_start = 0
train_end = int(np.floor(0.7*n))
test_start=train_end
test_end=n
data_train = df[train_start: train_end,:]
data_test = df[test_start:test_end,:]
#data_train = df_cut[train_start: train_end,:]
#data_test = df_cut[test_start:test_end,:]

#Scale data
scaler = MinMaxScaler()
scaler.fit(data_train)
data_train = scaler.transform(data_train)
data_test = scaler.transform(data_test)
# Build X and y
x_train = data_train[:, 1:]
y_train = data_train[:, 0]
x_test = data_test[:, 1:]
y_test = data_test[:, 0]


# Initializers
sigma = 1
weight_initializer = tf.variance_scaling_initializer(mode="fan_avg", distribution="uniform", scale=sigma)
bias_initializer = tf.zeros_initializer()



# Model architecture parameters
n_related = 4
n_neurons_1 = 1024
n_neurons_2 = 512
n_neurons_3 = 256
n_neurons_4 = 128
n_target = 1
# Layer 1: Variables for hidden weights and biases
W_hidden_1 = tf.Variable(weight_initializer([n_related, n_neurons_1]))
bias_hidden_1 = tf.Variable(bias_initializer([n_neurons_1]))
# Layer 2: Variables for hidden weights and biases
W_hidden_2 = tf.Variable(weight_initializer([n_neurons_1, n_neurons_2]))
bias_hidden_2 = tf.Variable(bias_initializer([n_neurons_2]))
# Layer 3: Variables for hidden weights and biases
W_hidden_3 = tf.Variable(weight_initializer([n_neurons_2, n_neurons_3]))
bias_hidden_3 = tf.Variable(bias_initializer([n_neurons_3]))
# Layer 4: Variables for hidden weights and biases
W_hidden_4 = tf.Variable(weight_initializer([n_neurons_3, n_neurons_4]))
bias_hidden_4 = tf.Variable(bias_initializer([n_neurons_4]))

# Output layer: Variables for output weights and biases
W_out = tf.Variable(weight_initializer([n_neurons_4, n_target]))
bias_out = tf.Variable(bias_initializer([n_target]))


# Placeholder
X = tf.placeholder(dtype=tf.float32, shape=[None, n_related])
Y = tf.placeholder(dtype=tf.float32, shape=[None])

# Hidden layer
hidden_1 = tf.nn.relu(tf.add(tf.matmul(X, W_hidden_1), bias_hidden_1))
hidden_2 = tf.nn.relu(tf.add(tf.matmul(hidden_1, W_hidden_2), bias_hidden_2))
hidden_3 = tf.nn.relu(tf.add(tf.matmul(hidden_2, W_hidden_3), bias_hidden_3))
hidden_4 = tf.nn.relu(tf.add(tf.matmul(hidden_3, W_hidden_4), bias_hidden_4))


# Output layer (must be transposed)
out = tf.transpose(tf.add(tf.matmul(hidden_4, W_out), bias_out))
# Cost function
mse = tf.reduce_mean(tf.squared_difference(out, Y))
#   set the learning rate for optimizer 
learning_rate = 0.02

#   use the loss function (Mean squared error)
tf_error = tf.reduce_sum(tf.pow(y_test-Y, 2))/(2*0.8*n)

# Optimizer
opt = tf.train.AdamOptimizer().minimize(mse)
#opt = tf.train.GradientDescentOptimizer(learning_rate).minimize(tf_error)

# Session
session = tf.InteractiveSession()

# Init
session.run(tf.global_variables_initializer())

# Setup plot
plt.ion()
fig = plt.figure()
ax1 = fig.add_subplot(111)
line1, = ax1.plot(y_test)
line2, = ax1.plot(y_test * 0.5)
plt.show()

# Fit neural net
batch_size = 128
mse_train = []
mse_test = []

# Run
epochs = 50
for e in range(epochs):

    # Shuffle training data
    shuffle_indices = np.random.permutation(np.arange(len(y_train)))
    x_train = x_train[shuffle_indices]
    y_train = y_train[shuffle_indices]

    # Minibatch training
    for i in range(0, len(y_train) // batch_size):
        start = i * batch_size
        batch_x = x_train[start:start + batch_size]
        batch_y = y_train[start:start + batch_size]
        # Run optimizer with batch
        session.run(opt, feed_dict={X: batch_x, Y: batch_y})

        # Show progress
        if np.mod(i, 10) == 0:
            # MSE train and test
            mse_train.append(session.run(mse, feed_dict={X: x_train, Y: y_train}))
            mse_test.append(session.run(mse, feed_dict={X: x_test, Y: y_test}))
            print('MSE Train: ', mse_train[-1])
            print('MSE Test: ', mse_test[-1])
            # Prediction
            pred = session.run(out, feed_dict={X: x_test})
            line2.set_ydata(pred)
            plt.title('Epoch ' + str(e) + ', Batch ' + str(i))
            plt.pause(0.01)
plt.show(block=True)

# Print final MSE after Training
mse_final = session.run(mse, feed_dict={X: x_test, Y: y_test})
print(mse_final)





































