from target import Target
import os
import numpy as np
import utilities as ut
import tensorflow as tf
import math
from random import shuffle

class CarvanaTarget(Target):
    def __init__(self, path, sample, crop_images = True, num_pixels=None, scale=1.0):
        self.path = path
        self.sample = sample
        self.crop_images = crop_images
        self.img_shape = ut.get_image_shape(self.sample, crop=self.crop_images,scale=scale) if num_pixels == None else num_pixels

    def generator(self, tensors, path, crop,scale):
        ret = []
        try:
            for fn, l in tensors:
                img = ut.read_image(path=path, fname=fn, show=False, crop=crop, scale=scale)
                #print('One image:'+fn)
                ret.append((img,l))

        except TypeError as te:
            #print("Got a TypeError, tensors are {tensors}, path is {path}, crop is {crop}, scale is {scale}".format(tensors=tensors, path=path, crop=crop, scale=scale))
            return []

        return ret

    def get_tensor_list(self,  path, num_classes=16, num = None, onehot=False):
        files = os.listdir(path)

        if not files:
            return None

        jpgs = [f for f in files if f.endswith('jpg') or f.endswith('jpeg')] # this gets abcd132d_37.jpg
        number_in_filename = [name_fragment.split('_')[1] for name_fragment in jpgs] # this gets 37.jpg
        number_in_filename = [name_fragment.split('.')[0] for name_fragment in number_in_filename] # this gets 37
        label_array = np.asarray(number_in_filename, dtype=np.int32) - 1
        if onehot == True:
            labels = np.zeros((len(label_array), num_classes))
            labels[np.arange(len(label_array)), label_array] = 1
        else:
            labels = label_array

        if num is None:
            num = len(jpgs)

        # return must be list of tuples (filename, label array [one-hot bool])
 #       self.tensorlist = list(zip(jpgs[:num], labels[:num]))
        ret = list(zip(jpgs[:num], labels[:num]))
        shuffle(ret)
        return ret

    def get_graph_placeholders(self, img_shape=None,num_classes=16, batch_size=10):
        if img_shape == None:
            img_shape = self.img_shape
        pixel_num = img_shape[0] * img_shape[1] * img_shape[2]
#what about type here?
        images_placeholder = tf.placeholder(tf.float32, shape=(batch_size, pixel_num), name='Images')
        labels_placeholder = tf.placeholder(tf.int32, shape=(batch_size), name='Labels')

        return (images_placeholder, labels_placeholder)

    def inference(self, images_placeholder, hidden1_units, hidden2_units, num_classes = 16, img_shape=None):
        if img_shape == None:
            img_shape = self.img_shape
        pixel_num = img_shape[0] * img_shape[1] * img_shape[2]
        with tf.name_scope('inference'):
            with tf.name_scope('hidden1'):
                weights = tf.truncated_normal([pixel_num, hidden1_units],
                                              stddev=1.0/math.sqrt(float(pixel_num)),
                                              name='weights')
                biases = tf.Variable(tf.zeros([hidden1_units]),
                                     name='biases')

                hidden1 = tf.nn.relu(tf.matmul(images_placeholder, weights) + biases)

            with tf.name_scope('hidden2'):
                weights = tf.truncated_normal([hidden1_units, hidden2_units],
                                              stddev=1.0 / math.sqrt(float(pixel_num)),
                                              name='weights')
                biases = tf.Variable(tf.zeros([hidden2_units]),
                                     name='biases')

                hidden2 = tf.nn.relu(tf.matmul(hidden1, weights) + biases)

            with tf.name_scope('softmax_linear'):
                weights = tf.truncated_normal([hidden2_units, num_classes],
                                              stddev=1.0 / math.sqrt(float(hidden2_units)),
                                              name='weights')
                biases = tf.Variable(tf.zeros([num_classes]),
                                     name='biases')

                logits = tf.matmul(hidden2, weights) + biases

                return logits


    def loss(self,logits,labels):
        with tf.name_scope('loser'):
            labels=tf.to_int64(labels)
            cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(
                labels=labels, logits=logits, name='xentropy')

            return  tf.reduce_mean(cross_entropy, name='xentropy_mean')

    def evaluation(self, logits, labels):
        with tf.name_scope('evaluation'):
            correct = tf.nn.in_top_k(logits,labels,4,name='correct_evaluation')
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

    def do_eval(self, sess, eval_op, pl_imgs, pl_labels, tensor_list, batch_size, data_path, crop, scale):
        true_count = 0
        steps_per_epoch = len(tensor_list) // batch_size
        num_examples = steps_per_epoch * batch_size
        for step in range(steps_per_epoch):
            for tensors in ut.grouper(tensor_list, batch_size):
                if tensors is None or len(tensors) < batch_size:
                    break
                tensor_batch=self.generator(tensors, path=data_path, crop=crop,scale=scale)
                if len(tensor_batch) < batch_size:
                    break
                imgs = [tupl[0] for tupl in tensor_batch]
                labels = [tupl[1] for tupl in tensor_batch]
                true_count + sess.run(eval_op, feed_dict = {pl_imgs:imgs, pl_labels:labels})

            precision = float(true_count)/num_examples
            print('Num examples: %d  Num correct: %d  Precision @ 1: %0.04f' %
                  (num_examples, true_count, precision))


