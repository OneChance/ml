import tensorflow as tf
import numpy as np

# W = tf.Variable([[1, 2, 3], [4, 5, 6]], dtype=tf.float32, name='weights')
# b = tf.Variable([[1, 2, 3]], dtype=tf.float32, name='biases')
#
# init = tf.global_variables_initializer()
#
# saver = tf.train.Saver()
#
# with tf.Session() as session:
#     session.run(init)
#     save_path = saver.save(session, "train_result/test.ckpt")
#     print("save to:", save_path)

# restore
W = tf.Variable(np.arange(6).reshape((2, 3)), dtype=tf.float32, name="weights")
b = tf.Variable(np.arange(3).reshape((1, 3)), dtype=tf.float32, name="biases")

saver = tf.train.Saver()

with tf.Session() as session:
    saver.restore(session, "train_result/test.ckpt")
    print("weights:", session.run(W))
    print("biases:", session.run(b))
