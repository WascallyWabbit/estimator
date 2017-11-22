#use pipelines & estimator to handle carvana images (or iceberg)

# set up tensor graph
# set up logging & tensorboard
# make list of tensors w/file names for each of test, train, validate
# in pipeline, read files into arrays, set up labels prn, scramble prn

# <editor-fold desc="Main">
import utilities as ut
from carvana_target import CarvanaTarget
import tensorflow as tf
import time
import os
import sys

def main():
    # <editor-fold desc="Process parameters into globals">
    FLAGS, unparsed = ut.parseArgs()
    # TEST_DATA_PATH      = FLAGS.test_data_path
    SAMPLE_FILE = FLAGS.train_data_path + FLAGS.sample
    IMG_SHAPE   = ut.get_image_shape(crop=FLAGS.crop, filename=SAMPLE_FILE,scale=FLAGS.scale)

    if FLAGS.target == 'carvana':
        target = CarvanaTarget(path=FLAGS.train_data_path,sample=SAMPLE_FILE,crop_images=FLAGS.crop,scale=FLAGS.scale)
    else:
        return

# set up graph
    with tf.Graph().as_default():
        (image_placeholder, label_placeholder) = target.get_graph_placeholders(img_shape=IMG_SHAPE, batch_size=FLAGS.batch_size)
        logits_op = target.inference(images_placeholder=image_placeholder,hidden1_units=FLAGS.hidden1_units,hidden2_units=FLAGS.hidden2_units)
        evaluation_op = target.evaluation(logits=logits_op, labels=label_placeholder)
        loss_op = target.loss(logits=logits_op, labels=label_placeholder)
        train_op = target.training(learning_rate=FLAGS.learning_rate, loss_op=loss_op)

        # Create a saver for writing training checkpoints.
        saver = tf.train.Saver()

        sess = tf.InteractiveSession(config = tf.ConfigProto(log_device_placement=True))

        # Build the summary Tensor based on the TF collection of Summaries.
        summary = tf.summary.merge_all()

        # Instantiate a SummaryWriter to output summaries and the Graph.
        summary_writer = tf.summary.FileWriter(FLAGS.tb_dir, sess.graph)

        gvi=tf.global_variables_initializer()
        gvi.run(session=sess)
        start_time = time.time()

#chop list into chunx, process each one
        for epoch in range(FLAGS.epochs):
            print('Epoch {epoch} of {epochs}'.format(epoch=epoch+1, epochs=FLAGS.epochs))
            epoch_start_time = time.time()
            # get tensor lists
            # get train, test, validate(?) lists
            tensor_list_train = target.get_tensor_list(path=FLAGS.train_data_path)
            tensor_list_test = target.get_tensor_list(path=FLAGS.test_data_path)
            batch_num = 0
            loss=0.0
            batches = len(tensor_list_train) // FLAGS.batch_size
            for tensors in ut.grouper(tensor_list_train,FLAGS.batch_size):
                batch_num += 1
                #sys.stdout.write('Batch {batch_num} of {batches} batches.'.format(batch_num=batch_num, batches=batches))
                batch_start_time = time.time()
                if tensors is None or len(tensors) < FLAGS.batch_size:
                    break
                tensor_batch=target.generator(tensors,path=FLAGS.train_data_path, crop=FLAGS.crop,scale=FLAGS.scale)
                if tensor_batch == []:
                    break
                imgs = [tupl[0] for tupl in tensor_batch]
                labels = [tupl[1] for tupl in tensor_batch]

                _, loss=sess.run([train_op, loss_op], feed_dict={image_placeholder:imgs, label_placeholder:labels})
                print('Loss:%f, batch elapsed time %.3f, batch %d of %d'% (loss, time.time() - batch_start_time, batch_num, 1+(len(tensor_list_train)//FLAGS.batch_size)))

            epoch_total_time = time.time() - epoch_start_time
            print('Epoch time:%.3f secs'% (epoch_total_time))
            duration = time.time() - start_time
            if epoch % 3 == 0:
                # Print status to stdout.
                print('Step %d: loss = %.2f (%.3f sec)' % (epoch, loss, duration))
                # Update the events file.
                summary_str = sess.run(summary, feed_dict={image_placeholder:imgs, label_placeholder:labels})
                summary_writer.add_summary(summary_str, epoch)
                summary_writer.flush()

            if (epoch + 1) % 10 == 0 or (epoch + 1) == FLAGS.epochs:
                print('Training Data Eval:')
                target.do_eval(sess=sess,
                               eval_op=evaluation_op,
                               pl_imgs=image_placeholder,
                               pl_labels=label_placeholder,
                               tensor_list=tensor_list_train,
                               batch_size=FLAGS.batch_size,
                               data_path=FLAGS.train_data_path,
                               crop=FLAGS.crop,
                               scale=FLAGS.scale)
                checkpoint_file = os.path.join(FLAGS.tb_dir, 'model.ckpt')
                saver.save(sess, checkpoint_file, global_step=epoch)
                # Evaluate against the training set.

                print('Test Data Eval:')
                target.do_eval(sess=sess,
                               eval_op=evaluation_op,
                               pl_imgs=image_placeholder,
                               pl_labels=label_placeholder,
                               tensor_list=tensor_list_test,
                               batch_size=FLAGS.batch_size,
                               data_path=FLAGS.test_data_path,
                               crop=FLAGS.crop,
                               scale=FLAGS.scale)

    pass

if __name__ == '__main__':
    main()
# </editor-fold>
