from __future__ import print_function

import tensorflow as tf
import numpy as np
from tensorflow.contrib import rnn

tf.set_random_seed(777)  # reproducibility 
sentence = ("if you want to build a ship, don't drum up people together to "
            "collect wood and don't assign them tasks and work, but rather "
            "teach them to long for the endless immensity of the sea.")

char_set = list(set(sentence))
char_dic = {w: i for i, w in enumerate(char_set)}

data_dim = len(char_set)
hidden_size = len(char_set)
num_classes = len(char_set)
sequence_length = 10  # Any arbitrary number
learning_rate = 0.1

print(char_set)
print(char_dic)

dataX = []
dataY = []

for i in range(0, len(sentence) - sequence_length):
    x_str = sentence[i:i + sequence_length]
    y_str = sentence[i+1:i + sequence_length+1]
    print(i, x_str, '->', y_str)

    x = [char_dic[c] for c in x_str]
    y = [char_dic[c] for c in y_str]
    print(x,y)

    dataX.append(x)
    dataY.append(y)

batch_size = len(dataX)

X = tf.placeholder(tf.int32, [None, sequence_length])
Y = tf.placeholder(tf.int32, [None, sequence_length])

X_one_hot =tf.one_hot(X, num_classes)
print(X_one_hot)

cell = tf.contrib.rnn.BasicLSTMCell(num_units = hidden_size, state_is_tuple = True)
cell = rnn.MultiRNNCell([cell]*4, state_is_tuple = True)

initial_state = cell.zero_state(batch_size, tf.float32)
outputs, _states = tf.nn.dynamic_rnn(cell, X_one_hot, initial_state=initial_state, dtype = tf.float32)

weight = tf.ones([batch_size, sequence_length])
#RNN에서 나온 outputs을 바로 sequence_loss - logits에 넣으면 좋지 않다. softmax로 변환 한 뒤에 loss를 구하는 것이 
X_for_softmax = tf.reshape(outputs, [-1, hidden_size])
softmax_w =tf.get_variable("softmax w", [hidden_size,num_classes])
softmax_b =tf.get_variable("softmax b", [num_classes])
outputs = tf.matmul(X_for_softmax, softmax_w) + softmax_b
outputs = tf.reshape(outputs, [batch_size, sequence_length, num_classes])

sequence_loss = tf.contrib.seq2seq.sequence_loss(logits=outputs, targets = Y, weights = weight)
loss = tf.reduce_mean(sequence_loss)
train = tf.train.AdamOptimizer(learning_rate=0.1).minimize(loss)

prediction = tf.argmax(outputs, axis = 2)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for i in range(2000):
        l, _ = sess.run([loss, train], feed_dict={X: dataX, Y: dataY})
        result = sess.run(prediction, feed_dict={X: dataX})
