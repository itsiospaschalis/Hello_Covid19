# -*- coding: utf-8 -*-
import tensorflow.compat.v1 as tf
sess = tf.Session()

  
with tf.Session() as sess:
  new_saver = tf.train.import_meta_graph("C:/Users/itsios/Desktop/dissertation/checkpoints/model.ckpt-1495066.meta")
  new_saver.restore(sess,tf.train.latest_checkpoint("C:/Users/itsios/Desktop/dissertation/checkpoints/model.ckpt-1495066"))  
