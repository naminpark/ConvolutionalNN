# reviser: Park, Nam In

from __future__ import print_function

import tensorflow as tf

# Import MNIST data
from tensorflow.examples.tutorials.mnist import input_data


class CNN():

    training_iters = 200000
    batch_size = 128
    display_step = 10

    def __init__(self,lr = 0.001, n_input=784,n_classes=10, dropout=0.75,weight='he',bias='zero'):
        #Parameters
        self.learning_rate = 0.001

        # Network Parameters
        self.n_input = n_input # MNIST data input (img shape: 28*28)
        self.n_classes = n_classes # MNIST total classes (0-9 digits)
        self.dropout = dropout # Dropout, probability to keep units
        self.setVariable()

        self.weights = self.weight_initializer[weight]
        self.biases = self.bias_initializer[bias]
        self.Model()
        # Initializing the variables
        init = tf.global_variables_initializer()
        self.sess=tf.Session()
        self.sess.run(init)



    def setVariable(self):

        # tf Graph input
        self.x = tf.placeholder(tf.float32, [None, self.n_input])
        self.y = tf.placeholder(tf.float32, [None, self.n_classes])
        self.keep_prob = tf.placeholder(tf.float32) #dropout (keep probability)

        # Store layers weight & bias
        self.normal_weights = {
            # 5x5 conv, 1 input, 32 outputs
            'wc1': tf.Variable(tf.random_normal([5, 5, 1, 32])),
            # 5x5 conv, 32 inputs, 64 outputs
            'wc2': tf.Variable(tf.random_normal([5, 5, 32, 64])),
            # fully connected, 7*7*64 inputs, 1024 outputs
            'wd1': tf.Variable(tf.random_normal([7*7*64, 1024])),
            # 1024 inputs, 10 outputs (class prediction)
            'out': tf.Variable(tf.random_normal([1024, self.n_classes]))
        }

        self.truncated_normal_weights = {
            # 5x5 conv, 1 input, 32 outputs
            'wc1': tf.Variable(tf.random_normal([5, 5, 1, 32])),
            # 5x5 conv, 32 inputs, 64 outputs
            'wc2': tf.Variable(tf.random_normal([5, 5, 32, 64])),
            # fully connected, 7*7*64 inputs, 1024 outputs
            'wd1': tf.Variable(tf.random_normal([7*7*64, 1024])),
            # 1024 inputs, 10 outputs (class prediction)
            'out': tf.Variable(tf.random_normal([1024, self.n_classes]))
        }


        self.xavier_weights = {
            # 5x5 conv, 1 input, 32 outputs
            'wc1': tf.get_variable('wc1_xaiver',[5, 5, 1, 32], initializer=tf.contrib.layers.xavier_initializer()),
            # 5x5 conv, 32 inputs, 64 outputs
            'wc2': tf.get_variable('wc2_xaiver',[5, 5, 32, 64], initializer=tf.contrib.layers.xavier_initializer()),
            # fully connected, 7*7*64 inputs, 1024 outputs
            'wd1': tf.get_variable('wd1_xaiver',[7*7*64, 1024], initializer=tf.contrib.layers.xavier_initializer()),
            # 1024 inputs, 10 outputs (class prediction)
            'out': tf.get_variable('out_xaiver',[1024, self.n_classes], initializer=tf.contrib.layers.xavier_initializer()),
        }

        self.he_weights = {
            # 5x5 conv, 1 input, 32 outputs
            'wc1': tf.get_variable('wc1_he',[5, 5, 1, 32], initializer=tf.contrib.layers.variance_scaling_initializer()),
            # 5x5 conv, 32 inputs, 64 outputs
            'wc2': tf.get_variable('wc2_he',[5, 5, 32, 64], initializer=tf.contrib.layers.variance_scaling_initializer()),
            # fully connected, 7*7*64 inputs, 1024 outputs
            'wd1': tf.get_variable('wd1_he',[7*7*64, 1024], initializer=tf.contrib.layers.variance_scaling_initializer()),
            # 1024 inputs, 10 outputs (class prediction)
            'out': tf.get_variable('out_he',[1024, self.n_classes], initializer=tf.contrib.layers.variance_scaling_initializer()),
        }


        self.normal_biases = {
            'bc1': tf.Variable(tf.random_normal([32])),
            'bc2': tf.Variable(tf.random_normal([64])),
            'bd1': tf.Variable(tf.random_normal([1024])),
            'out': tf.Variable(tf.random_normal([self.n_classes]))
        }

        self.zero_biases = {
            'bc1': tf.Variable(tf.zeros([32])),
            'bc2': tf.Variable(tf.zeros([64])),
            'bd1': tf.Variable(tf.zeros([1024])),
            'out': tf.Variable(tf.zeros([self.n_classes]))
        }

        self.weight_initializer = {'normal':self.normal_weights, 'truncated_normal':self.truncated_normal_weights, 'xavier':self.xavier_weights, 'he':self.he_weights}
        self.bias_initializer = {'normal':self.normal_biases, 'zero':self.zero_biases}





    # Create some wrappers for simplicity
    def conv2d(self,x, W, b, strides=1):
        # Conv2D wrapper, with bias and relu activation
        x = tf.nn.conv2d(x, W, strides=[1, strides, strides, 1], padding='SAME')
        x = tf.nn.bias_add(x, b)
        return tf.nn.relu(x)



    # Create model
    def conv_net(self,x, weights, biases, dropout):
        # Reshape input picture
        x = tf.reshape(x, shape=[-1, 28, 28, 1])

        # Convolution Layer
        conv1 = self.conv2d(x, weights['wc1'], biases['bc1'])
        # Max Pooling (down-sampling)
        conv1 = tf.nn.max_pool(conv1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

        # Convolution Layer
        conv2 = self.conv2d(conv1, weights['wc2'], biases['bc2'])
        # Max Pooling (down-sampling)
        conv2 = tf.nn.max_pool(conv2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

        # Fully connected layer
        # Reshape conv2 output to fit fully connected layer input
        fc1 = tf.reshape(conv2, [-1, weights['wd1'].get_shape().as_list()[0]])
        fc1 = tf.add(tf.matmul(fc1, weights['wd1']), biases['bd1'])
        fc1 = tf.nn.relu(fc1)
        # Apply Dropout
        fc1 = tf.nn.dropout(fc1, dropout)

        # Output, class prediction
        out = tf.add(tf.matmul(fc1, weights['out']), biases['out'])
        return out

    def Model(self):

        # Construct model
        pred = self.conv_net(self.x, self.weights, self.biases, self.keep_prob)

        # Define loss and optimizer
        self.cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=pred, labels=self.y))
        self.optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate).minimize(self.cost)

        # Evaluate model
        correct_pred = tf.equal(tf.argmax(pred, 1), tf.argmax(self.y, 1))
        self.accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))



    def RUN(self,mnist):

        step = 1
        # Keep training until reach max iterations
        while step * self.batch_size < self.training_iters:
            batch_x, batch_y = mnist.train.next_batch(self.batch_size)
            # Run optimization op (backprop)
            self.sess.run(self.optimizer, feed_dict={self.x: batch_x, self.y: batch_y,
                                           self.keep_prob: self.dropout})
            if step % self.display_step == 0:
                # Calculate batch loss and accuracy
                loss, acc = self.sess.run([self.cost, self.accuracy], feed_dict={self.x: batch_x,
                                                                  self.y: batch_y,
                                                                  self.keep_prob: 1.})
                print("Iter " + str(step*self.batch_size) + ", Minibatch Loss= " + \
                      "{:.6f}".format(loss) + ", Training Accuracy= " + \
                      "{:.5f}".format(acc))
            step += 1
        print("Optimization Finished!")

        # Calculate accuracy for 256 mnist test images
        print("Testing Accuracy:", \
            self.sess.run(self.accuracy, feed_dict={self.x: mnist.test.images[:256],
                                          self.y: mnist.test.labels[:256],
                                          self.keep_prob: 1.}))




if __name__ == "__main__":

    cnn=CNN()

    mnist = input_data.read_data_sets("MNIST_data", one_hot=True)

    cnn.RUN(mnist)
