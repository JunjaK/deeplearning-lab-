import tensorflow as tf
import numpy as np

sample = "if you want you"
idx2char = list(set(sample))
char2idx = {c: i for i, c in enumerate(idx2char)}
sample_idx = [char2idx[c] for c in sample]

dic_size = len(char2idx)
rnn_hidden_size = len(char2idx)
sequence_length = len(sample) - 1
batch_size = 1
num_classes = len(char2idx)

x_data = [sample_idx[:-1]]
y_data = [sample_idx[1:]]

             
X = tf.placeholder(tf.int32, [None, sequence_length])
Y = tf.placeholder(tf.int32, [None, sequence_length])

X_one_hot =tf.one_hot(X, num_classes)
print(X_one_hot)

cell = tf.contrib.rnn.BasicLSTMCell(num_units = rnn_hidden_size, state_is_tuple = True)
initial_state = cell.zero_state(batch_size, tf.float32)
outputs, _states = tf.nn.dynamic_rnn(cell, X_one_hot, initial_state=initial_state, dtype = tf.float32)

weight = tf.ones([batch_size, sequence_length])
#RNN에서 나온 outputs을 바로 sequence_loss - logits에 넣으면 좋지 않다.
sequence_loss = tf.contrib.seq2seq.sequence_loss(logits=outputs, targets = Y, weights = weight)
loss = tf.reduce_mean(sequence_loss)
train = tf.train.AdamOptimizer(learning_rate=0.1).minimize(loss)

prediction = tf.argmax(outputs, axis = 2)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for i in range(2000):
        l, _ = sess.run([loss, train], feed_dict={X: x_data, Y: y_data})
        result = sess.run(prediction, feed_dict={X: x_data})
        print(i, "loss:", l, "prediction:", result, "true Y", y_data)

        result_str = [idx2char[c] for c in np.squeeze(result)]
        print("\tPrediction str:", ''.join(result_str))
