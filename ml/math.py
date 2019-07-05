# -*- coding: utf-8 -*-
"""
Created on Fri May 25 18:04:35 2018

@author: Myron
"""

import tensorflow as tf

x=tf.Variable (2, name="X")
y=tf.Variable (3, name="Y")
f=x*x+x*y
g=x*(x+y)

with tf.Session() as sess:
 sess.run(tf.global_variables_initializer())
 fwrd_res=f.eval()
 print ("fwrd_res=",fwrd_res)
 grad1=tf.gradients(f,[x,y])
 grad2=tf.gradients(g,[x,y])
 #print("grad1_ops=",grad1[0]," ",grad1[1])
 print("grad1_vals=",grad1[0].eval()," ",grad1[1].eval())
 print("grad2_vals=",grad2[0].eval()," ",grad2[1].eval())
 grad3=tf.gradients(f,tf.global_variables())
 grad4=tf.gradients(g,tf.global_variables())
 #print("grad2_ops=",grad2[0]," ",grad2[1])
 print("grad3_vals=",grad3[0].eval()," ",grad3[1].eval())
 print("grad4_vals=",grad4[0].eval()," ",grad4[1].eval())