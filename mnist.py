from model import OSELM
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

# Basic tf setting
tf.set_random_seed(2016)
sess = tf.Session()

# Get data
mnist = input_data.read_data_sets("/tmp/data/", one_hot=True)

# Construct ELM
batch_size = 1100
hidden_num = 1000
print("batch_size : {}".format(batch_size))
print("hidden_num : {}".format(hidden_num))

elm = OSELM(sess, batch_size, 784, hidden_num, 10)

# one-step feed-forward training
k = batch_size
while k <= mnist.train.num_examples:
  #print("batch : {}".format(k))
  train_x, train_y = mnist.train.next_batch(batch_size)
  elm.train(train_x, train_y)
  k += batch_size

# testing
elm.test(mnist.test.images, mnist.test.labels)
