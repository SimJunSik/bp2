import os
import numpy as np
from numpy import array
import tensorflow as tf

train_input = np.load("train_data.npy")
train_label = np.load("train_label.npy")

test_input = np.load("test_data.npy")
test_label = np.load("test_label.npy")

# hyper parameters
learning_rate = 0.001

width = 128
height = width
num_of_class = 2
channel = 3

# input place holders
X = tf.placeholder(tf.float32, shape = [None, width*height*channel])
X_img = tf.reshape(X, [-1, width, height, channel])
Y = tf.placeholder(tf.float32, [None, num_of_class])
keep_prob = tf.placeholder(tf.float32)


W1 = tf.Variable(tf.random_normal([3, 3, channel, 32], stddev=5e-2))
L1 = tf.nn.conv2d(X_img, W1, strides=[1, 1, 1, 1], padding='SAME')
L1 = tf.nn.relu(L1)
L1 = tf.nn.dropout(L1, keep_prob=keep_prob)
L1 = tf.nn.max_pool(L1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

W2 = tf.Variable(tf.random_normal([3, 3, 32, 64], stddev=5e-2))
L2 = tf.nn.conv2d(L1, W2, strides=[1, 1, 1, 1], padding='SAME')
L2 = tf.nn.relu(L2)
L2 = tf.nn.dropout(L2, keep_prob=keep_prob)
L2 = tf.nn.max_pool(L2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')


W3 = tf.Variable(tf.random_normal([3, 3, 64, 128], stddev=5e-2))
L3 = tf.nn.conv2d(L2, W3, strides=[1, 1, 1, 1], padding='SAME')
L3 = tf.nn.relu(L3)
L3 = tf.nn.dropout(L3, keep_prob=keep_prob)
L3 = tf.nn.max_pool(L3, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')


W4 = tf.Variable(tf.random_normal([3, 3, 128, 128], stddev=5e-2))
L4 = tf.nn.conv2d(L3, W4, strides=[1, 1, 1, 1], padding='SAME')
L4 = tf.nn.relu(L4)
L4 = tf.nn.dropout(L4, keep_prob=keep_prob)



#stddev = 0.01
W5 = tf.Variable(tf.random_normal([3, 3, 128, 128], stddev=5e-2))
L5 = tf.nn.conv2d(L4, W5, strides=[1, 1, 1, 1], padding='SAME')
L5 = tf.nn.relu(L5)
L5 = tf.nn.dropout(L5, keep_prob=keep_prob)


W6 = tf.Variable(tf.random_normal([3, 3, 128, 256], stddev=5e-2))
L6 = tf.nn.conv2d(L5, W6, strides=[1, 1, 1, 1], padding='SAME')
L6 = tf.nn.relu(L6)
L6 = tf.nn.dropout(L6, keep_prob=keep_prob)



L6_flat = tf.reshape(L6, [-1, (width//8) * (height//8) * 256])
W7 = tf.get_variable("W7", shape=[(width//8) * (height//8) * 256, 384], initializer=tf.contrib.layers.xavier_initializer())
b = tf.Variable(tf.random_normal([384]))
L7 = tf.nn.relu(tf.matmul(L6_flat, W7) + b)
L7_drop = tf.nn.dropout(L7, keep_prob)


L7_flat = tf.reshape(L7, [-1, 384])
W8 = tf.get_variable("W8", shape=[384, num_of_class], initializer=tf.contrib.layers.xavier_initializer())
b = tf.Variable(tf.random_normal([num_of_class]))


logits = tf.matmul(L7_flat,W8) + b
y_pred = tf.nn.softmax(logits)


# define cost/loss &amp;amp;amp; optimizer
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
    logits=logits, labels=Y))
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

training_epochs = 100
batch_size = 40


#save
saver = tf.train.Saver()


# initialize
sess = tf.Session()
sess.run(tf.global_variables_initializer())

#accuracy
correct_prediction = tf.equal(tf.argmax(y_pred, 1), tf.argmax(Y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

# train my model
print('Learning started. It takes sometime.')
for epoch in range(training_epochs):
    avg_cost = 0
    total_batch = int(len(train_input) / batch_size)

    for i in range(total_batch):
        start = ((i + 1) * batch_size) - batch_size
        end = ((i + 1) * batch_size)
        batch_xs = train_input[start:end]
        batch_ys = train_label[start:end]
        feed_dict = {X: batch_xs, Y: batch_ys, keep_prob : 0.8}
        c, _ = sess.run([cost, optimizer], feed_dict=feed_dict)
        avg_cost += c / total_batch
        #train_accuracy = accuracy.eval(feed_dict={X: batch_xs, Y: batch_ys, keep_prob : 1})
        saver.save(sess, "./model/model.ckpt")

    print('Epoch:', '%04d' % (epoch + 1), 'cost =', '{:.9f}'.format(avg_cost), end=' ')
    print('train accuracy = ',sess.run(accuracy, feed_dict={X: batch_xs, Y: batch_ys, keep_prob : 1}))
print('Learning Finished!')

# Test model and check accuracy
correct_prediction = tf.equal(tf.argmax(logits, 1), tf.argmax(Y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
print('Accuracy:', sess.run(accuracy, feed_dict={
    X: test_input, Y: test_label, keep_prob : 1}))
