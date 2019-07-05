# -*- coding: utf-8 -*-
"""
Created on Sun Apr 29 15:53:43 2018

@author: Myron
"""


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler

# Import data
data = pd.read_csv('data_stocks.csv')
# Drop date variable
data = data.drop(['DATE'], 1)
# Dimensions of dataset
n = data.shape[0]
p = data.shape[1]
# Make data a numpy array
data1 = data.values
# Plot SP500 vs Time
plt.plot(data['SP500'])

# Training and test data
train_start = 0
train_end = int(np.floor(0.8*n))
test_start = train_end
test_end = n
data_train = data1[train_start: train_end, :]
data_test = data1[test_start: test_end, :]

# Scale data

scaler = MinMaxScaler()
scaler.fit(data_train)
data_train = scaler.transform(data_train)
data_test = scaler.transform(data_test)
# Build X and y
x=data1[:,1:]
y=data1[:,0]
X_train = data_train[:, 1:]
y_train = data_train[:, 0]
X_test = data_test[:, 1:]
y_test = data_test[:, 0]

# Variable 
n_stocks = 500
layer_1 =1024
layer_2 =512
layer_3 =256
layer_4 =128
target_layer =1

# Session
net = tf.InteractiveSession()

# Placeholder
X = tf.placeholder(dtype=tf.float32, shape=[None,n_stocks])
Y = tf.placeholder(dtype=tf.float32, shape=[None])

# Initializers
sigma = 1
weight_initializer = tf.variance_scaling_initializer(mode="fan_avg", distribution="uniform", scale=sigma)
bias_initializer = tf.zeros_initializer()


tf_x_factor = tf.Variable(np.random.randn(), name="x_factor")
tf_y_offset = tf.Variable(np.random.randn(), name="y_offset")

# Weights & Biases
# Layer 1
#weight_1 = tf.Variable(weight_initializer([n_stocks,layer_1]))

# Layer 1: Variables for hidden weights and biases
W_hidden_1 = tf.Variable(weight_initializer([n_stocks, layer_1]))
bias_hidden_1 = tf.Variable(bias_initializer([layer_1]))
# Layer 2: Variables for hidden weights and biases
W_hidden_2 = tf.Variable(weight_initializer([layer_1, layer_2]))
bias_hidden_2 = tf.Variable(bias_initializer([layer_2]))
# Layer 3: Variables for hidden weights and biases
W_hidden_3 = tf.Variable(weight_initializer([layer_2, layer_3]))
bias_hidden_3 = tf.Variable(bias_initializer([layer_3]))
# Layer 4: Variables for hidden weights and biases
W_hidden_4 = tf.Variable(weight_initializer([layer_3, layer_4]))
bias_hidden_4 = tf.Variable(bias_initializer([layer_4]))

# Output layer: Variables for output weights and biases
W_out = tf.Variable(weight_initializer([layer_4, target_layer]))
bias_out = tf.Variable(bias_initializer([target_layer]))

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



# Init
net.run(tf.global_variables_initializer())

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
    X_train = X_train[shuffle_indices]
    y_train = y_train[shuffle_indices]

    # Minibatch training
    for i in range(0, len(y_train) // batch_size):
        start = i * batch_size
        batch_x = X_train[start:start + batch_size]
        batch_y = y_train[start:start + batch_size]
        # Run optimizer with batch
        net.run(opt, feed_dict={X: batch_x, Y: batch_y})

        # Show progress
        if np.mod(i, 50) == 0:
            # MSE train and test
            mse_train.append(net.run(mse, feed_dict={X: X_train, Y: y_train}))
            mse_test.append(net.run(mse, feed_dict={X: X_test, Y: y_test}))
            print('MSE Train: ', mse_train[-1])
            print('MSE Test: ', mse_test[-1])
            # Prediction
            pred = net.run(out, feed_dict={X: X_test})
            line2.set_ydata(pred)
            plt.title('Epoch ' + str(e) + ', Batch ' + str(i))
            plt.pause(0.01)
plt.show(block=True)
            
# initializing
#init = tf.global_variables_initializer()
            
## start session
#with tf.Session() as sess:
#    sess.run(init)
#
#    # set how often to display training progress and number of training iterations
#    display_every = 1
#    num_training_iter = 200
#
#    # calculate the number of lines to animation
#    fit_num_plots = math.floor(num_training_iter/display_every)
#    # add storage of factor and offset values from each epoch
#    fit_x_factor = np.zeros(fit_num_plots)
#    fit_y_offsets = np.zeros(fit_num_plots)
#    fit_plot_idx = 0    
#
#   # keep iterating the training data
#    for iteration in range(num_training_iter):
#
##        # Fit all training data
##        for (x, y) in zip(X_train, y_train):
###            sess.run(opt, feed_dict={X: X_train, Y: data_test})
##            sess.run(opt, feed_dict={X: x, Y: y})
#            
#        batch_size = 256   
#        for i in range(0, len(y_train) // batch_size):
#                start = i * batch_size
#                batch_x = X_train[start:start + batch_size]
#                batch_y = y_train[start:start + batch_size]
#                # Run optimizer with batch
#                sess.run(opt, feed_dict={X: batch_x, Y: batch_y})            
#            
#            
#            
#            
#
#        # Display current status
#        if (iteration + 1) % display_every == 0:
#            c = sess.run(mse, feed_dict={X: X_train, Y:y_train})
#            print("iteration:", '%04d' % (iteration + 1), "error=", "{:.9f}".format(c), \
#                "x_factor=", sess.run(tf_x_factor), "y_offset=", sess.run(tf_y_offset))
#            # Save the fit size_factor and price_offset to allow animation of learning process
#            fit_x_factor[fit_plot_idx] = sess.run(tf_x_factor)
#            fit_y_offsets[fit_plot_idx] = sess.run(tf_y_offset)
#            fit_plot_idx = fit_plot_idx + 1
#
#    print("Optimization Done")
#    training_cost = sess.run(mse, feed_dict={X: X_train, Y:y_train})
#    print("Trained error=", training_cost, "x_factor=", sess.run(tf_x_factor), "y_offset=", sess.run(tf_y_offset), '\n')
#
#   # Plot of training and test data, and learned regression
#    
#    # get values used to normalized data so we can denormalize data back to its original scale
#    train_x_mean = X_train.mean()
#    train_x_std = X_train.std()
#
#    train_y_mean = y_train.mean()
#    train_y_std = y_train.std()
#
#    # Plot the graph
#    plt.rcParams["figure.figsize"] = (10,8)
#    plt.figure()
#    plt.ylabel("y")
#    plt.xlabel("x")
##    plt.plot(X_train, y_train, 'go', label='Training data')
#    plt.plot(X_test, y_test, 'mo', label='Testing data')
#    plt.plot(X_train * train_x_std + train_x_mean,
#             (sess.run(tf_x_factor) * X_train + sess.run(tf_y_offset)) * train_y_std + train_y_mean,
#             label='Learned Regression')
# 
#    plt.legend(loc='upper left')
#    plt.show()
            
            
   
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            