import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data as mnist_data

tf.set_random_seed(0)

mnist = mnist_data.read_data_sets("data", one_hot=True, reshape=False, validation_size=0)

x = tf.placeholder(tf.float32,[None, 28, 28, 1])
#w = tf.Variable(tf.zeros([784, 10]))
#b = tf.Variable(tf.zeros([10]))

w1 = tf.Variable(tf.truncated_normal([28*28, 200] ,stddev=0.1))
b1 = tf.Variable(tf.zeros([200]))
w2 = tf.Variable(tf.truncated_normal([200, 100], stddev=0.1))
b2 = tf.Variable(tf.zeros([100]))
w3 = tf.Variable(tf.truncated_normal([100, 60], stddev=0.1))
b3 = tf.Variable(tf.zeros([60]))
w4 = tf.Variable(tf.truncated_normal([60, 30], stddev=0.1))
b4 = tf.Variable(tf.zeros([30]))
w5 = tf.Variable(tf.truncated_normal([30, 10], stddev=0.1))
b5 = tf.Variable(tf.zeros([10]))

#init = tf.initialize_all_variables()

xx = tf.reshape(x, [-1, 28*28])

y1 = tf.nn.sigmoid(tf.matmul(xx, w1) + b1)
y2 = tf.nn.sigmoid(tf.matmul(y1, w2) + b2)
y3 = tf.nn.sigmoid(tf.matmul(y2, w3) + b3)
y4 = tf.nn.sigmoid(tf.matmul(y3, w4) + b4)
ylogits = tf.matmul(y4, w5) + b5
y = tf.nn.softmax(ylogits)

y_ = tf.placeholder(tf.float32,[None, 10])

#cross_entropy = -tf.reduce_sum(y_ * tf.log(y))
cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=ylogits, labels=y_)
cross_entropy = tf.reduce_mean(cross_entropy)*100
#cross_entropy = tf.reduce_sum(cross_entropy)

is_correct = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
accuracy = tf.reduce_mean(tf.cast(is_correct, tf.float32))

#optimizer = tf.train.GradientDescentOptimizer(0.003)
optimizer = tf.train.AdamOptimizer(0.003)
train_step = optimizer.minimize(cross_entropy)

init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init)

for i in range(10000):
    batch_X, batch_Y = mnist.train.next_batch(100)
    train_data={x: batch_X, y_: batch_Y}

    sess.run(train_step, feed_dict = train_data)

#a,c = sess.run([accuracy, cross_entropy], feed_dict = train_data)
#print("Train Data Accuracy: {0}".format(a))

test_data={x: mnist.test.images, y_: mnist.test.labels}
a,c = sess.run([accuracy, cross_entropy], feed_dict = test_data)
print("Test Data Accuracy: {0}".format(a))
