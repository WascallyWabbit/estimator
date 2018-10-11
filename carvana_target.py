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
        label_array = np.asarray(number_in_filename, dtype=np.int32) - 1
        if onehot == True:
            labels = np.zeros((len(label_array), num_classes))
            labels[np.arange(len(label_array)), label_array] = 1
        else:
            labels = label_array

        if num is None:
            num = len(jpgs)

        # return must be list of tuples (filename, label array [one-hot bool])
        ret = list(zip(jpgs[:num], labels[:num]))
        shuffle(ret)
        return ret


    def init_weights(self,pixel_num,hidden1_units,hidden2_units,num_classes):
        w1 = tf.Variable(
            tf.truncated_normal([pixel_num, hidden1_units],
                                      stddev=1.0 / math.sqrt(float(pixel_num)),
                                      name='weights1')
        )

        w2 = tf.Variable(
            tf.truncated_normal([hidden1_units, hidden2_units],
                                      stddev=1.0 / math.sqrt(float(pixel_num)),
                                      name='weights2')
        )

        w3 = tf.Variable(
            tf.truncated_normal([hidden2_units, num_classes],
                                      stddev=1.0 / math.sqrt(float(hidden2_units)),
                                      name='weights3')
        )

        b1 = tf.Variable(tf.zeros([hidden1_units]),
                              name='biases1')

        b2 = tf.Variable(tf.zeros([hidden2_units]),
                              name='biases2')

        b3 = tf.Variable(tf.zeros([num_classes]),
                              name='biases3')

        return (w1,w2,w3,b1,b2,b3)

    def get_graph_placeholders(self, img_shape=None, batch_size=10, num_classes=16):
        if img_shape == None:
            img_shape = self.img_shape

        pixel_num = ut.pixnum_from_img_shape(img_shape)

        images_placeholder = tf.placeholder(tf.float32, shape=(batch_size, pixel_num), name='Images')
        labels_placeholder = tf.placeholder(tf.float32, shape=(batch_size, num_classes), name='Labels')

        return (images_placeholder, labels_placeholder)

    def inference(self,
                  images_placeholder,
                  hidden1_units,
                  hidden2_units,
                  w1,w2,w3,
                  b1,b2,b3,
                  num_classes = 16,
                  img_shape=None):

        if img_shape == None:
            img_shape = self.img_shape
        pixel_num = ut.pixnum_from_img_shape(img_shape)


        with tf.name_scope('inference'):
            #       with tf.name_scope('inference'):
            display_tensor = tf.reshape(tensor=images_placeholder, shape=[64,img_shape[1],img_shape[0],1])
            tf.summary.image(tensor=display_tensor, max_outputs=320,name="Carvana_images")

            with tf.name_scope('hidden1'):
                #weights1 = tf.Variable(
                #    tf.truncated_normal([pixel_num, hidden1_units],
                #                              stddev=1.0/math.sqrt(float(pixel_num))),
                #                              name='weights1')
                #biases1 = tf.Variable(tf.zeros([hidden1_units]),
                #                     name='biases1')

                hidden1 = tf.nn.relu(tf.matmul(images_placeholder, w1) + b1)
                tf.summary.histogram(name='w1', values=w1)
                tf.summary.histogram(name='b1', values=b1)


            with tf.name_scope('hidden2'):
                #weights2 = tf.Variable(
                #    tf.truncated_normal([hidden1_units, hidden2_units],
                #                              stddev=1.0 / math.sqrt(float(hidden1_units))),
                #                              name='w2')
                #biases2 = tf.Variable(tf.zeros([hidden2_units]),
                #                     name='b2')

                hidden2 = tf.nn.relu(tf.matmul(hidden1, w2) + b2)
                tf.summary.histogram(name='w2', values=w2)
                tf.summary.histogram(name='b2', values=b2)

            with tf.name_scope('softmax_linear'):
                #w3 = tf.Variable(
                #    tf.truncated_normal([hidden2_units, num_classes],
                #                              stddev=1.0 / math.sqrt(float(hidden2_units))),
                #    name='w3')
                #biases3 = tf.Variable(tf.zeros([num_classes]),
                #                     name='b3')

                logits = tf.nn.softmax(tf.matmul(hidden2, w3) + b3)
                #logits = tf.matmul(hidden2, w3) + b3

                tf.summary.histogram(name='weights3', values=w3)
                tf.summary.histogram(name='biases3', values=b3)
                tf.summary.histogram(name='logits', values=logits)

            return logits

    def loss(self,logits,labels):
        with tf.name_scope('loser'):
            #labels=tf.to_int64(labels)
            cross_entropy = tf.losses.softmax_cross_entropy(
                onehot_labels=labels, logits=logits)

            # cross_entropy = tf.nn.l2_loss(labels-logits)

            # rm = tf.reduce_mean(cross_entropy, name='xentropy_mean')
            # tf.summary.scalar('xentropy_reduced_mean', rm)
            return cross_entropy

    def evaluation(self, logits, labels):
        with tf.name_scope('evaluation'):
            #rs = tf.reduce_sum(tf.cast(correct,tf.int32), name='Reduce_sum')
            #tf.summary.scalar('Reduced sum', rs)
            #return correct
            return labels, logits

    def training(self, loss_op, learning_rate):
        with tf.name_scope('training'):
            tf.summary.scalar('Training loss_op', loss_op)
            optimizer = tf.train.AdamOptimizer(learning_rate, name='Adam_Optimizer')
            global_step = tf.Variable(0, name='global_step', trainable=False)
            tf.summary.scalar('Training global_step', global_step)
            train_op = optimizer.minimize(loss_op, global_step=global_step)
            return train_op
