import  argparse

def parseArgs():
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_data_path', type=str,
                        default='/Users/Eric Fowler/Downloads/carvana/train/',
                        help='Directory for storing input data')
    # parser.add_argument('--model', type=str,
    #                     default='none',
    #                     help='Model for processing data')
    parser.add_argument('--test_data_path', type=str,
                        default='/Users/Eric Fowler/Downloads/carvana/test/',
                        help='Directory for storing input data')
    parser.add_argument('--target', type=str,
                        default='mnist',
                        choices=['carvana', 'mnist'],
                        help='MNIST or Carvana?')
    # parser.add_argument('--minimize', type=str,
    #                     default='simple',
    #                     choices=['simple', 'cross'],
    #                     help='Simple or X-entropy?')
    # parser.add_argument('--train_step', type=str,
    #                     default='sgd',
    #                     choices=['sgd', 'adam'],
    #                     help='SGD or Adam optimization?')
    # parser.add_argument('--env', type=str,
    #                     default='pc',
    #                     choices=['pc', 'aws'],
    #                     help='pc or aws?')
    parser.add_argument('--sample', type=str,
                        default='0cdf5b5d0ce1_01.jpg',
                        help='Sample image file for sizing feature tensor')
    parser.add_argument('--numclasses', type=int,
                        default=16,
                        help='Carvana=16, MNIST=10')
    parser.add_argument('--learning_rate', type=float,
                        default=0.25,
                        help='Learning rate')
    parser.add_argument('--hidden1_units', type=int,
                        default=128,
                        help='Hidden layer #1 units')
    parser.add_argument('--hidden2_units', type=int,
                        default=32,
                        help='Hidden layer #2 units')
    # parser.add_argument('--num_test_images', type=int,
    #                     default=200000,
    #                     help='Number of images to test')
    # parser.add_argument('--num_train_images', type=int,
    #                     default=200000,
    #                     help='Number of images to train')
    # parser.add_argument('--crop', type=bool,
    #                     default=False,
    #                     help='Crop images for speed?')
    parser.add_argument('--show', type=bool,
                        default=False,
                        help='Show some images?')
    parser.add_argument('--scale', type=float,
                        default=1.0,
                        help='Scaling factor for images')
    parser.add_argument('--epochs', type=int,
                        default=1,
                        help='Epochs')
    parser.add_argument('--batch_size', type=int,
                        default=20,
                        help='Cut samples into chunks of this size')
    # parser.add_argument('--test_csv', type=str,
    #                     default='testout.csv',
    #                     help='File and path for storing test output file')
    parser.add_argument('--tb_dir', type=str,
                        default='/Users/eric fowler/tensorlog/',
                        help='Directory For Tensorboard log')
    parser.add_argument('--process', type=str,
                        default='shi-tomasi',
                        help='Edge finding process, \'shi-tomasi\' or \'harris\'')

    return parser.parse_known_args()


from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import cv2

def scale_image(img, scale=1.0):
    if scale != 1.0:
        img = np.array(img)
        s = img.shape
        img = cv2.resize(src=img, dsize=(int(s[1] // scale), int(s[0] // scale)))
    return img

def flatten_image(img):
    img = img.flatten('F')
    return img

def get_image_shape(filename, scale, show=False):
    img = read_image(filename=filename, show=show)
    img = scale_image(img=img, scale=scale)

    return img.shape

def pixnum_from_img_shape(img_shape):
    pixel_num = 1
    for t in img_shape:
        pixel_num *= t

    return pixel_num

def read_image_split_path(path, fname, show):
    return read_image(filename=path + fname, show=show)

def read_image(filename, show):
   mm = cv2.imread(filename)
   mm = cv2.cvtColor(mm, cv2.COLOR_BGR2GRAY)

   if show == True:
       plt.imshow(mm)
       plt.show()

#   mm = mm.convert('F')

   return mm

def find_edges(img, process):
    if process=='shi-tomasi':
        #corners = cv2.goodFeaturesToTrack(img, maxCorners=2500, qualityLevel=0.001, minDistance=2, mask=-1)
        corners = cv2.goodFeaturesToTrack(img, 2500, 0.001, 2)
        corners = np.int0(corners)
        img = np.full_like(img, 255)

        for i in corners:
            x, y = i.ravel()
            cv2.circle(img, (x, y), 10, (0, 0, 0), -1)

    elif process == 'harris':
        corners = cv2.cornerHarris(img, 8, 3, 0.01)
        x = corners.max()
        img[np.abs(corners) > 0.01 * corners.max()] = [255, 0, 0]
        img1 = img[:, :, 0]
        img = img1 - corners

    return img


import itertools as it
import sys

if sys.version[0]=='2':
    it.zip_longest=it.izip_longest

def grouper(iterable, n, fillvalue=None):
    "Collect data into fixed-length chunks or blocks"
    # grouper('ABCDEFG', 3, 'x') --> ABC DEF Gxx"
    a = [iter(iterable)]*n
    ret = it.zip_longest(*a, fillvalue=fillvalue)
    args= [n for n in ret if n is not None]

    return args