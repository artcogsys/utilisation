import tensorflow as tf
import os

from PIL import Image

# from internal_logger import logger
from model import ADEResNet
import numpy as np

from tf_record_writer import _get_image_filenames

CHECKPOINT_FOLDER = 'checkpoints'
CHECKPOINT_NAME = 'PVANET'
CHECKPOINT_STEP = 20

flags = tf.app.flags
FLAGS = flags.FLAGS

flags.DEFINE_integer('num_classes', 151, 'Size of each training batch')


class Evaluate:
    def __init__(self, num_output_classes):
        self.num_output_classes = num_output_classes
        self.model = ADEResNet(batch_size=1,
                               image_size=(None, None),
                               num_output_classes=self.num_output_classes,
                               placeholder_inputs=True)

        self.sess = tf.Session(graph=self.model.graph)
        self.logger = get_logger()

    def evaluate(self, image_dir, annotation_dir, split_name):
        assert split_name in ['training', 'validation']

        image_dir = os.path.join(image_dir, split_name)
        annotation_dir = os.path.join(annotation_dir, split_name)

        filenames = zip(_get_image_filenames(image_dir),
                        _get_image_filenames(annotation_dir))
        # All matching files must have same name
        assert all([x[:-4] == y[:-4] for x, y in filenames])

        with self.model.graph.as_default():
            # self.sess.run(tf.variables_initializer(tf.global_variables()))
            saver = tf.train.Saver()
            latest_checkpoint = tf.train.latest_checkpoint(CHECKPOINT_FOLDER)
            self.sess.run(tf.variables_initializer(tf.local_variables()))
            self.log("loading from checkpoint file: " + latest_checkpoint)
            saver.restore(self.sess, latest_checkpoint)

            losses = np.zeros([2000])
            ious = np.zeros([2000])
            count = 0

            for i in range(0, 2000):
                image_filename, label_filename = filenames[i]
                image_filename = os.path.join(image_dir, image_filename)
                label_filename = os.path.join(annotation_dir, label_filename)

                image = Image.open(image_filename)
                label = Image.open(label_filename)

                dim_1 = image.size[0]
                dim_2 = image.size[1]

                # image, label = standard_size_with_ratio(image, label)
                image = np.expand_dims(np.array(image).astype(np.float32) / 256, 0)
                label = np.expand_dims(np.array(label), 0)
                competition_loss, mean_iou, = self.sess.run(
                    [self.model.competition_loss, self.model.mean_iou], feed_dict={self.model.input: image,
                                                                                   self.model.truth: label})
                count += 1
                self.log(
                    "%4d - comp: %.3f, iou: %.3f, dimensions: %d*%d" % (i, competition_loss, mean_iou, dim_1, dim_2))
                losses[i] = competition_loss
                ious[i] = mean_iou

            self.log("mean loss: %.3f" % np.mean(losses))
            self.log("mean iou: %.3f" % np.mean(ious))
            self.sess.close()

    def log(self, message):
        print message
        self.logger.info(message)


def run(dataset_dir):
    image_dir = os.path.join(dataset_dir, 'images')
    annotation_dir = os.path.join(dataset_dir, 'annotations')

    e = Evaluate(num_output_classes=151)
    e.evaluate(image_dir, annotation_dir, 'validation')

    print('\nFinished converting the Ade20k dataset!')


def get_logger():
    import logging

    logger = logging.getLogger('utilization_classifier')
    logger.setLevel(logging.DEBUG)
    fh = logging.FileHandler('evaluate.log')
    fh.setLevel(logging.DEBUG)
    formatter = logging.Formatter('%(asctime)s - %(message)s')
    fh.setFormatter(formatter)
    # add the handlers to the logger
    logger.addHandler(fh)
    return logger


def keep_original_size(image, label):
    # dim_1 = label.size[0]
    # if dim_1 % 8 != 0:
    #     dim_1 = dim_1 - (dim_1 % 8) + 8
    #
    # dim_2 = label.size[1]
    # if dim_2 % 8 != 0:
    #     dim_2 = dim_2 - (dim_2 % 8) + 8
    #
    # image = image.crop((dim_1 - image.size[0], dim_2 - image.size[1], dim_1, dim_2))
    # label = label.crop((dim_1 - image.size[0], dim_2 - image.size[1], dim_1, dim_2))
    # label = label.resize((dim_1 / 8, dim_2 / 8), Image.NEAREST)

    return image, label


def standard_size_with_ratio(image, label):
    # size = 256
    # shape = np.array([label.size[0], label.size[1]])
    # ratio = float(size) / np.max(shape)
    # shape = (shape * ratio).astype(np.int32)
    #
    # dim_1 = shape[0]
    # if dim_1 % 8 != 0:
    #     dim_1 = dim_1 - (dim_1 % 8) + 8
    #
    # dim_2 = shape[1]
    # if dim_2 % 8 != 0:
    #     dim_2 = dim_2 - (dim_2 % 8) + 8
    #
    # shape = np.array([dim_1, dim_2])
    # # if ratio > 1:
    # #     image = image.resize(shape, Image.BILINEAR)
    # # if ratio < 1:
    # image = image.resize(shape, Image.BILINEAR)
    #
    # label = label.resize(shape / 8, Image.NEAREST)

    return image, label


run('./data')
