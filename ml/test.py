# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

#   import
import tensorflow as tf

sess = tf.Session()

#   print string
hello = tf.constant('Hello, TensorFlow!')
print(sess.run(hello))

#   addition
a=tf.constant(32)
b=tf.constant(21)
print('a+b ={0}'.format(sess.run(a+b)))




