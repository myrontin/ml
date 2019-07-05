# -*- coding: utf-8 -*-
"""
Created on Sun Mar  4 12:17:10 2018

@author: Myron
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# Imports
import numpy as np
import tensorflow as tf

tf.logging.set_verbosity(tf.logging.INFO)

# Our application logic will be added here

if __name__ == "__main__":
  tf.app.run()
  
def cnn_model_fn(features,labels,mode):
    #input layer
    input_layer=tf.reshape(features["x"],[-1,28,28,1])
    
    #convolutional layer #1
    