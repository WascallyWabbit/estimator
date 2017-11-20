#use pipelines & estimator to handle carvana images (or iceberg)

# set up tensor graph
# set up logging & tensorboard
# make list of tensors w/file names for each of test, train, validate
# in pipeline, read files into arrays, set up labels prn, scramble prn

# <editor-fold desc="Main">
import utilities as ut
from target import Target
from carvana_target import CarvanaTarget
import tensorflow as tf


def main():
    # <editor-fold desc="Process parameters into globals">
    FLAGS, unparsed = ut.parseArgs()
    # TEST_DATA_PATH      = FLAGS.test_data_path
    SAMPLE_FILE = FLAGS.train_data_path + FLAGS.sample
    IMG_SHAPE   = ut.get_image_shape(crop=FLAGS.crop, filename=SAMPLE_FILE)

    if FLAGS.target == 'carvana':
        target = CarvanaTarget(path=FLAGS.train_data_path,sample=SAMPLE_FILE,crop_images=FLAGS.crop)
    else:
        return


# set up graph
    with tf.Graph().as_default():
        (image_placeholder, label_placeholder) = target.get_graph_placeholders(img_shape=IMG_SHAPE, batch_size=FLAGS.batch_size)
        logits_op = target.inference(images_placeholder=image_placeholder,hidden1_units=FLAGS.hidden1_units,hidden2_units=FLAGS.hidden2_units)
        evaluation_op = target.evaluation(logits=logits_op, labels=label_placeholder)
        loss_op = target.loss(logits=logits_op, labels=label_placeholder)
        train_op = target.training(learning_rate=FLAGS.learning_rate, loss=loss_op)
        sess = tf.InteractiveSession(config = tf.ConfigProto(log_device_placement=True))
        gvi=tf.global_variables_initializer()
        gvi.run(session=sess)


#chop list into chunx, process each one
        for epoch in range(FLAGS.epochs):
            # get tensor lists
            # get train, test, validate(?) lists
            tensor_list_train = target.get_tensor_list(path=FLAGS.train_data_path)
            tensor_list_test = target.get_tensor_list(path=FLAGS.test_data_path)

            for tensors in ut.grouper(tensor_list_train,FLAGS.batch_size):
                tensor_batch=target.generator(tensors,path=FLAGS.train_data_path)
                imgs = [tupl[0] for tupl in tensor_batch]
                labels = [tupl[1] for tupl in tensor_batch]

                _, loss=sess.run([train_op, loss_op], feed_dict={image_placeholder:imgs, label_placeholder:labels})
                print(loss)

    pass

if __name__ == '__main__':
    main()
# </editor-fold>
