from target import Target
import os
import numpy as np
import utilities as ut
import tensorflow as tf
import math
from random import shuffle
#--target carvana --numclasses 16 --train_data_path "/users/eric fowler/downloads/carvana/train/" --test_data_path "/users/eric fowler/downloads/carvana/test/"  --sample 0cdf5b5d0ce1_01.jpg  --batch_size 64 --tb_dir  "\Users\Eric Fowler\tensorlog\mine" --learning_rate 0.001 --epochs 2
class CarvanaTarget(Target):

    def get_tensor_list(self,  path, num_classes=16, num = None, onehot=False):
        files = os.listdir(path)

        if not files:
            return None

        jpgs = [f for f in files if f.endswith('jpg') or f.endswith('jpeg')] # this gets 'filename_37.jpg'
        number_in_filename = [name_fragment.split('_')[1] for name_fragment in jpgs] # this gets '37.jpg'
        number_in_filename = [name_fragment.split('.')[0] for name_fragment in number_in_filename] # this gets '37'
        #label_array = np.asarray(number_in_filename, dtype=np.int32) - 1
        if onehot == True:
            label_array = np.asarray(number_in_filename, dtype=np.int32) - 1
            labels = np.zeros((len(label_array), num_classes))
            labels[np.arange(len(label_array)), label_array] = 1
        else:
            label_array = np.asarray(number_in_filename, dtype=np.int32)
            labels = label_array

        if num is None:
            num = len(jpgs)

        # return must be list of tuples (filename, label array [one-hot bool])
 #       self.tensorlist = list(zip(jpgs[:num], labels[:num]))
        ret = list(zip(jpgs[:num], labels[:num]))
        shuffle(ret)
        return ret


    def init_weights(self,pixel_num,hidden1_units,hidden2_units,num_classes):
        w1 = tf.Variable(
            tf.truncated_normal([pixel_num, hidden1_units],
                                      stddev=1.0 / math.sqrt(float(pixel_num)),
                                      name='weights')
        )

        w2 = tf.Variable(
            tf.truncated_normal([hidden1_units, hidden2_units],
                                      stddev=1.0 / math.sqrt(float(pixel_num)),
                                      name='weights')
        )

        wl = tf.Variable(
            tf.truncated_normal([hidden2_units, num_classes],
                                      stddev=1.0 / math.sqrt(float(hidden2_units)),
                                      name='weights')
        )

        return (w1,w2,wl)

    def get_graph_placeholders(self, img_shape=None, batch_size=10, num_classes=16):
        if img_shape == None:
            img_shape = self.img_shape

        pixel_num = ut.pixnum_from_img_shape(img_shape)

        # what about type here?
        images_placeholder = tf.placeholder(tf.float32, shape=(batch_size, pixel_num), name='Images')
        labels_placeholder = tf.placeholder(tf.int32, shape=(batch_size), name='Labels')

        return (images_placeholder, labels_placeholder)

    def inference(self,
                  images_placeholder,
                  hidden1_units,
                  hidden2_units,
                  num_classes = 16,
                  img_shape=None):

        if img_shape == None:
            img_shape = self.img_shape
        pixel_num = ut.pixnum_from_img_shape(img_shape)


 #       with tf.name_scope('inference'):
            #tf.summary.image(tensor=images_placeholder, max_outputs=3,name="Carvana_images")
        with tf.name_scope('hidden1'):
            weights = tf.Variable(
                tf.truncated_normal([pixel_num, hidden1_units],
                                          stddev=1.0/math.sqrt(float(pixel_num))),
                                          name='weights')
            biases = tf.Variable(tf.zeros([hidden1_units]),
                                 name='biases')

            hidden1 = tf.nn.relu(tf.matmul(images_placeholder, weights) + biases)


        with tf.name_scope('hidden2'):
            weights = tf.Variable(
                tf.truncated_normal([hidden1_units, hidden2_units],
                                          stddev=1.0 / math.sqrt(float(hidden1_units))),
                                          name='weights')
            biases = tf.Variable(tf.zeros([hidden2_units]),
                                 name='biases')

            hidden2 = tf.nn.relu(tf.matmul(hidden1, weights) + biases)


        with tf.name_scope('softmax_linear'):
            weights = tf.Variable(
                tf.truncated_normal([hidden2_units, num_classes],
                                          stddev=1.0 / math.sqrt(float(hidden2_units))),
                name='weights')
            biases = tf.Variable(tf.zeros([num_classes]),
                                 name='biases')

            logits = tf.nn.softmax(tf.matmul(hidden2, weights) + biases)

            # tf.summary.histogram(name='weights', values=weights)
            # tf.summary.histogram(name='biases', values=biases)
            # tf.summary.histogram(name='logits', values=logits)

            return logits


    def loss(self,logits,labels):
#        with tf.name_scope('loser'):
        labels=tf.to_int64(labels)
        cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(
            labels=labels, logits=logits, name='xentropy')

        rm = tf.reduce_mean(cross_entropy, name='xentropy_mean')
        tf.summary.scalar('xentropy_reduced_mean', rm)
        return rm

    def evaluation(self, logits, labels):
        with tf.name_scope('evaluation'):
            correct = tf.nn.in_top_k(logits,labels,1,name='correct_evaluation')
#            tf.summary.scalar('Evaluation', correct)
            rs = tf.reduce_sum(tf.cast(correct,tf.int32), name='Reduce_sum')
            tf.summary.scalar('Reduced sum', rs)
            return rs

    def training(self, loss_op, learning_rate):
        with tf.name_scope('training'):
            tf.summary.scalar('Training loss_op', loss_op)
            optimizer = tf.train.GradientDescentOptimizer(learning_rate, name='Gradient_Descent_Optimizificator')
            global_step = tf.Variable(0, name='global_step', trainable=False)
            tf.summary.scalar('Training global_step', global_step)
            train_op = optimizer.minimize(loss_op, global_step=global_step)
            return train_op




