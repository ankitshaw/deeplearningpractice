import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data as mnist_data

tf.set_random_seed(0)

mnist = mnist_data.read_data_sets("data", one_hot=True, reshape=False, validation_size=0)

x = tf.placeholder(tf.float32,[None, 28, 28, 1])
w = tf.Variable(tf.zeros([784, 10]))
b = tf.Variable(tf.zeros([10]))

#init = tf.initialize_all_variables()
init = tf.global_variables_initializer()

y = tf.nn.softmax(tf.matmul(tf.reshape(x, [-1,784]), w) + b)
y_ = tf.placeholder(tf.float32,[None, 10])

cross_entropy = -tf.reduce_sum(y_ * tf.log(y))

is_correct = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
accuracy = tf.reduce_mean(tf.cast(is_correct, tf.float32))

optimizer = tf.train.GradientDescentOptimizer(0.003)
train_step = optimizer.minimize(cross_entropy)

sess = tf.Session()
sess.run(init)

for i in range(1000):
    batch_X, batch_Y = mnist.train.next_batch(100)
    train_data={x: batch_X, y_: batch_Y}

    sess.run(train_step, feed_dict = train_data)

#a,c = sess.run([accuracy, cross_entropy], feed_dict = train_data)
#print("Train Data Accuracy: {0}".format(a))

test_data={x: mnist.test.images, y_: mnist.test.labels}
a,c = sess.run([accuracy, cross_entropy], feed_dict = test_data)
print("Test Data Accuracy: {0}".format(a))
