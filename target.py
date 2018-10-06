import utilities as ut
import tensorflow as tf

class Target:
    def __init__(self, path, sample, num_pixels=None, scale=1.0):
        self.path = path
        self.sample = sample
        self.img_shape = ut.get_image_shape(filename=self.sample, scale=scale) if num_pixels == None else num_pixels


    def generator(self, tensors, path, scale, process):
        ret = []
        try:
            for fn, l in tensors:
                img = ut.read_image_split_path(path=path, fname=fn, show=False)
                img = ut.find_edges(img=img, process=process)
                img = ut.scale_image(img=img, scale=scale)
                img = ut.flatten_image(img=img)
                ret.append((img,l))

        except TypeError as te:
            print("Got a TypeError, scale is {scale}, message is {message}".format(tensors=tensors, path=path, message=te.args[0], scale=scale))
            return []

        return ret

    def do_eval(self, sess, eval_op, pl_imgs, pl_labels, tensor_list, batch_size, data_path, scale, process):
        steps_per_epoch = len(tensor_list) // batch_size
        num_examples = steps_per_epoch * batch_size
        #for step in range(steps_per_epoch):
        true_count = 0
        for tensors in ut.grouper(tensor_list, batch_size):
            if tensors is None or len(tensors) < batch_size:
                break
            tensor_batch = self.generator(tensors, path=data_path, scale=scale,process=process)
            if len(tensor_batch) < batch_size:
                break
            imgs = [tupl[0] for tupl in tensor_batch]
            labels = [tupl[1] for tupl in tensor_batch]
            true_count += sess.run(eval_op, feed_dict={pl_imgs: imgs, pl_labels: labels})

        precision = float(true_count) / num_examples
        print('Num examples: %d  Num correct: %d  Precision @ 1: %0.04f' %
              (num_examples, true_count, precision))