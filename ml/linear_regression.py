# -*- coding: utf-8 -*-
"""
Created on Sun Mar  4 20:31:56 2018

@author: Myron
"""

import tensorflow as tf
import numpy as np
import math
import matplotlib.pyplot as plt

#   generate random x 
num_of_data = 170
np.random.seed(52)
x= np.random.randint(low=0,high=1000,size=num_of_data)


#   generate random y
np.random.seed(52)
y= x*50 +np.random.randint(low=8000,high=20000,size=num_of_data)

#   plot the generated data
plt.plot(x, y,"bx")
plt.ylabel("y")
plt.xlabel("x")
plt.show()

#   normalize data
def normalize(array):
    return (array-array.mean())/array.std()

#   use 70% of the data as training set
num_train_samples=math.floor(num_of_data *0.7)

#   training data
train_x = np.asarray(x[:num_train_samples])
train_y = np.asarray(y[:num_train_samples])

train_norm_x = normalize(x)
train_norm_y = normalize(y)

#   test data
test_x = np.array(x[num_train_samples:])
test_y = np.array(y[num_train_samples:])

test_norm_x = normalize(test_x)
test_norm_y = normalize(test_y)

#   placeholders
tf_x = tf.placeholder("float", name="x")
tf_y = tf.placeholder("float", name="y")

#   x_factor & y_offset (stanard normal distribution)
tf_x_factor = tf.Variable(np.random.randn(), name="x_factor")
tf_y_offset = tf.Variable(np.random.randn(), name="y_offset")

# 2. Define the operations for the predicting values - predicted price = (size_factor * house_size ) + price_offset
#  Notice, the use of the tensorflow add and multiply functions.  These add the operations to the computation graph,
#  AND the tensorflow methods understand how to deal with Tensors.  Therefore do not try to use numpy or other library 
#  methods.
#   linear prediction y=mx+b (y=x_factor*x+y_offset )

tf_pred_y = tf.add(tf.multiply(tf_x_factor, tf_x), tf_y_offset)

#   use the loss function (Mean squared error)
#tf_error = tf.reduce_sum(tf.pow(tf_pred_y-tf_y, 2))/(2*num_train_samples)

#   use cross entropy loss
def CrossEntropy(tf_pred_y, tf_y):
    return tf.reduce_sum(-tf_y*tf.log(tf_pred_y))

tf_error=CrossEntropy(tf_pred_y,tf_y)


#   set the learning rate for optimizer 
learning_rate = 0.02

#   use gradient descent optimizer
optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(tf_error)

# initializing
init = tf.global_variables_initializer()

# start session
with tf.Session() as sess:
    sess.run(init)

    # set how often to display training progress and number of training iterations
    display_every = 1
    num_training_iter = 200

    # calculate the number of lines to animation
    fit_num_plots = math.floor(num_training_iter/display_every)
    # add storage of factor and offset values from each epoch
    fit_x_factor = np.zeros(fit_num_plots)
    fit_y_offsets = np.zeros(fit_num_plots)
    fit_plot_idx = 0    

   # keep iterating the training data
    for iteration in range(num_training_iter):

        # Fit all training data
        for (x, y) in zip(train_norm_x, train_norm_y):
            sess.run(optimizer, feed_dict={tf_x: x, tf_y: y})

        # Display current status
        if (iteration + 1) % display_every == 0:
            c = sess.run(tf_error, feed_dict={tf_x: train_norm_x, tf_y:train_norm_y})
            print("iteration:", '%04d' % (iteration + 1), "error=", "{:.9f}".format(c), \
                "x_factor=", sess.run(tf_x_factor), "y_offset=", sess.run(tf_y_offset))
            # Save the fit size_factor and price_offset to allow animation of learning process
            fit_x_factor[fit_plot_idx] = sess.run(tf_x_factor)
            fit_y_offsets[fit_plot_idx] = sess.run(tf_y_offset)
            fit_plot_idx = fit_plot_idx + 1

    print("Optimization Done")
    training_cost = sess.run(tf_error, feed_dict={tf_x: train_norm_x, tf_y: train_norm_y})
    print("Trained error=", training_cost, "x_factor=", sess.run(tf_x_factor), "y_offset=", sess.run(tf_y_offset), '\n')

   # Plot of training and test data, and learned regression
    
    # get values used to normalized data so we can denormalize data back to its original scale
    train_x_mean = train_x.mean()
    train_x_std = train_x.std()

    train_y_mean = train_y.mean()
    train_y_std = train_y.std()

    # Plot the graph
    plt.rcParams["figure.figsize"] = (10,8)
    plt.figure()
    plt.ylabel("y")
    plt.xlabel("x")
    plt.plot(train_x, train_y, 'go', label='Training data')
    plt.plot(test_x, test_y, 'mo', label='Testing data')
    plt.plot(train_norm_x * train_x_std + train_x_mean,
             (sess.run(tf_x_factor) * train_norm_x + sess.run(tf_y_offset)) * train_y_std + train_y_mean,
             label='Learned Regression')
 
    plt.legend(loc='upper left')
    plt.show()

























