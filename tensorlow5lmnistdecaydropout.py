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

#no of nodes in layers i, h1, h2, h3, h4, o
K, L, M, N, O, P = 784, 200, 100, 60, 30, 10

#Weights(random -0.2 to +0.2) and Biases(small +ve for activation function- RELU)
w1 = tf.Variable(tf.truncated_normal([K, L] ,stddev=0.1))
b1 = tf.Variable(tf.ones([L])/10)
w2 = tf.Variable(tf.truncated_normal([L, M], stddev=0.1))
b2 = tf.Variable(tf.ones([M])/10)
w3 = tf.Variable(tf.truncated_normal([M, N], stddev=0.1))
b3 = tf.Variable(tf.ones([N])/10)
w4 = tf.Variable(tf.truncated_normal([N, O], stddev=0.1))
b4 = tf.Variable(tf.ones([O])/10)
w5 = tf.Variable(tf.truncated_normal([O, P], stddev=0.1))
b5 = tf.Variable(tf.zeros([10])) #output layer uses sigmoid

#to flatten the 28x28 matrix to 1x784 matrix
xx = tf.reshape(x, [-1, 28*28])

#output at layers with dropout
y1 = tf.nn.relu(tf.matmul(xx, w1) + b1)
y1d = tf.nn.dropout(y1, pkeep)

y2 = tf.nn.relu(tf.matmul(y1d, w2) + b2)
y2d = tf.nn.dropout(y2, pkeep)

y3 = tf.nn.sigmoid(tf.matmul(y2d, w3) + b3)
y3d = tf.nn.dropout(y3, pkeep)

y4 = tf.nn.sigmoid(tf.matmul(y3d, w4) + b4)
y4d = tf.nn.dropout(y4, pkeep)

ylogits = tf.matmul(y4d, w5) + b5
y = tf.nn.softmax(ylogits)


#cross entropy loss function
cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=ylogits, labels=y_)
cross_entropy = tf.reduce_mean(cross_entropy)*100

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
