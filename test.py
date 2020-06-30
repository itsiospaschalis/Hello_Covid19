# -*- coding: utf-8 -*-
"""
Created on Tue Jun 30 23:52:08 2020

@author: itsios
"""


import tensorflow.compat.v1 as tf
###
sess = tf.Session()

# ή αυτο (1)
session=tf.compat.v1.Session()
saver = tf.train.import_meta_graph("C:\\Users\\itsios\\Desktop\\dissertation\\checkpoints\\model.ckpt-1495066.meta")
saver.restore(sess,tf.train.latest_checkpoint("C:\\Users\\itsios\\Desktop\\dissertation\\checkpoints\\model.ckpt-1495066.txt"))

#ή αυτο  (2)
with tf.Session() as sess:
  new_saver = tf.train.import_meta_graph("C:\\Users\\itsios\\Desktop\\dissertation\\checkpoints\\model.ckpt-1495066.meta")
  new_saver.restore(sess, tf.train.latest_checkpoint('C:\\Users\\itsios\\Desktop\\dissertation\\checkpoints\\model.ckpt-1495066.txt'))
