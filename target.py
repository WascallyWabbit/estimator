import utilities as ut
import tensorflow as tf

class Target:
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