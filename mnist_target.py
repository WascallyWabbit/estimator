from target import Target
import tensorflow as tf
import math
import os
import numpy as np
import utilities as ut
from random import shuffle

class MNISTTarget(Target):
    def init_weights(self, pixel_num, hidden1_units, hidden2_units, num_classes):
        w1 = tf.truncated_normal([pixel_num, hidden1_units],
                                 stddev=1.0 / math.sqrt(float(pixel_num)),
                                 name='weights')

        w2 = tf.truncated_normal([hidden1_units, hidden2_units],
                                 stddev=1.0 / math.sqrt(float(pixel_num)),
                                 name='weights')

        wl = tf.truncated_normal([hidden2_units, num_classes],
                                 stddev=1.0 / math.sqrt(float(hidden2_units)),
                                 name='weights')

        return (w1, w2, wl)

    def get_graph_placeholders(self, img_shape=None, batch_size=10, num_classes=10):
        if img_shape == None:
            img_shape = self.img_shape

        pixel_num = ut.pixnum_from_img_shape(img_shape)

        # what about type here?
        images_placeholder = tf.placeholder(tf.float32, shape=(batch_size, pixel_num), name='Images')
        labels_placeholder = tf.placeholder(tf.int32, shape=(batch_size), name='Labels')

        return (images_placeholder, labels_placeholder)

    def inference(self, images_placeholder, hidden1_units, hidden2_units, weights_tuple, num_classes=10,
                  img_shape=None):
        if img_shape == None:
            img_shape = self.img_shape

        w1 = weights_tuple[0]
        w2 = weights_tuple[1]
        wl = weights_tuple[2]
        with tf.name_scope('inference'):
            # tf.summary.image(tensor=images_placeholder, max_outputs=3,name="Carvana_images")
            with tf.name_scope('hidden1'):
                # weights = tf.truncated_normal([pixel_num, hidden1_units],
                #                               stddev=1.0/math.sqrt(float(pixel_num)),
                #                               name='weights')
                biases = tf.Variable(tf.zeros([hidden1_units]),
                                     name='biases')

                hidden1 = tf.nn.relu(tf.matmul(images_placeholder, w1) + biases)
                tf.summary.histogram(name='weights', values=w1)
                tf.summary.histogram(name='biases', values=biases)
                tf.summary.histogram(name='hidden1', values=hidden1)

            with tf.name_scope('hidden2'):
                # weights = tf.truncated_normal([hidden1_units, hidden2_units],
                #                               stddev=1.0 / math.sqrt(float(pixel_num)),
                #                               name='weights')
                biases = tf.Variable(tf.zeros([hidden2_units]),
                                     name='biases')

                hidden2 = tf.nn.relu(tf.matmul(hidden1, w2) + biases)
                tf.summary.histogram(name='weights', values=w2)
                tf.summary.histogram(name='biases', values=biases)
                tf.summary.histogram(name='hidden2', values=hidden2)

            with tf.name_scope('softmax_linear'):
                # weights = tf.truncated_normal([hidden2_units, num_classes],
                #                               stddev=1.0 / math.sqrt(float(hidden2_units)),
                #                               name='weights')
                biases = tf.Variable(tf.zeros([num_classes]),
                                     name='biases')

                logits = tf.nn.softmax(tf.matmul(hidden2, wl) + biases)

                tf.summary.histogram(name='weights', values=wl)
                tf.summary.histogram(name='biases', values=biases)
                tf.summary.histogram(name='logits', values=logits)

                return logits

    def evaluation(self, logits, labels):
        with tf.name_scope('evaluation'):
            #correct = tf.nn.in_top_k(logits,labels,4,name='correct_evaluation')
#            tf.summary.scalar('Evaluation', correct)
            rs = tf.reduce_sum(tf.cast(logits,tf.int32), name='Reduce_sum')
            tf.summary.scalar('Reduced sum', rs)
            return rs

    def loss(self,logits,labels):
        with tf.name_scope('loser'):
            labels=tf.to_int64(labels)
            cross_entropy = tf.nn.softmax_cross_entropy_with_logits(
                labels=labels, logits=logits, name='xentropy')

            rm = tf.reduce_mean(cross_entropy, name='xentropy_mean')
            tf.summary.scalar('xentropy_reduced_mean', rm)
            return rm

    def training(self, loss_op, learning_rate):
        with tf.name_scope('training'):
            tf.summary.scalar('Training loss_op', loss_op)
            optimizer = tf.train.GradientDescentOptimizer(learning_rate, name='Gradient_Descent_Optimizificator')
            global_step = tf.Variable(0, name='global_step', trainable=False)
            tf.summary.scalar('Training global_step', global_step)
            train_op = optimizer.minimize(loss_op, global_step=global_step)
            return train_op

    def get_tensor_list(self, path, num_classes=10, num = None, onehot=False):
        files = []
        labels = []
        for x in range(num_classes):
            label = np.zeros(num_classes)
            label[x] = 1
            if os.path.exists(path + str(x) + '/'):
                path_contains_digits = True
                fpath = path + str(x) + '/'
            else:
                path_contains_digits = False
                fpath = path
            jpgs = [f for f in os.listdir(fpath) if f.endswith('jpg') or f.endswith('jpeg')]
            for j in jpgs:
                if path_contains_digits == True:
                    files.append(str(x) + '/' + j)
                else:
                    files.append(j)
                    
                labels.append(label)

        if num == None:
            num = len(files)

        ret = list(zip(files[:num], labels[:num]))
        shuffle(ret)
        return ret

