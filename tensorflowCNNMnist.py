import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data as mnist_data
import math

tf.set_random_seed(0)

#lOAD DATA
mnist = mnist_data.read_data_sets("data", one_hot=True, reshape=False, validation_size=0)

#input images 28x28 size and its greyscale so last parameter is 1
x = tf.placeholder(tf.float32,[None, 28, 28, 1])
#true output
y_ = tf.placeholder(tf.float32,[None, 10])
#for variable learning rate
lr = tf.placeholder(tf.float32)
#for dropout
pkeep = tf.placeholder(tf.float32)

#layers
K = 6       # first convulational layer output depth
L = 12      # second CL
M = 24      # third CL
N = 200     # fully connected fourth layer

#weights and biases
w1 = tf.Variable(tf.truncated_normal([6, 6, 1, K] ,stddev=0.1))
b1 = tf.Variable(tf.constant(0.1, tf.float32, [K]))
w2 = tf.Variable(tf.truncated_normal([5, 5, K, L], stddev=0.1))
b2 = tf.Variable(tf.constant(0.1, tf.float32, [L]))
w3 = tf.Variable(tf.truncated_normal([4, 4, L, M], stddev=0.1))
b3 = tf.Variable(tf.constant(0.1, tf.float32, [M]))
w4 = tf.Variable(tf.truncated_normal([7*7*M, N], stddev=0.1))
b4 = tf.Variable(tf.constant(0.1, tf.float32, [N]))
w5 = tf.Variable(tf.truncated_normal([N, 10], stddev=0.1))
b5 = tf.Variable(tf.constant(0.1, tf.float32, [10]))

#model
stride = 1
y1 = tf.nn.relu(tf.nn.conv2d(x, w1, strides=[1, stride, stride, 1], padding='SAME') + b1)
stride = 2
y2 = tf.nn.relu(tf.nn.conv2d(y1, w2, strides=[1, stride, stride, 1], padding='SAME') + b2)
stride = 2
y3 = tf.nn.relu(tf.nn.conv2d(y2, w3, strides=[1, stride, stride, 1], padding='SAME') + b3)

yy = tf.reshape(y3, shape=[-1, 7*7*M])

y4 = tf.nn.relu(tf.matmul(yy, w4) + b4)
yy4 = tf.nn.dropout(y4, pkeep)

ylogits = tf.matmul(yy4, w5) + b5
y = tf.nn.softmax(ylogits)

#cross entropy loss function
cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=ylogits, labels=y_)
cross_entropy = tf.reduce_mean(cross_entropy)*100

#accuracy
is_correct = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
accuracy = tf.reduce_mean(tf.cast(is_correct, tf.float32))

#optimizer
optimizer = tf.train.AdamOptimizer(lr)
train_step = optimizer.minimize(cross_entropy)

#init
init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init)

#iterations
for i in range(10000):
    #hyper parameters
    max_lr = 0.003
    min_lr = 0.0001
    decay_speed = 2000.0
    learning_rate = min_lr + (max_lr - min_lr) * math.exp(-i/decay_speed)

    batch_X, batch_Y = mnist.train.next_batch(100)
    train_data={x: batch_X, y_: batch_Y, pkeep: 0.75, lr: learning_rate}

    sess.run(train_step, feed_dict = train_data)

#a,c = sess.run([accuracy, cross_entropy], feed_dict = train_data)
#print("Train Data Accuracy: {0}".format(a))

test_data={x: mnist.test.images, y_: mnist.test.labels, pkeep: 1.0}
a,c = sess.run([accuracy, cross_entropy], feed_dict = test_data)
print("Test Data Accuracy: {0}".format(a))

sess.close()
