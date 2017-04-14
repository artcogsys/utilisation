import tensorflow as tf

from class_embeddings import class_embedding_lookup_table
from internal_logger import logger
from model import PVANet
from settings import IMAGE_HEIGHT, IMAGE_WIDTH, BATCH_SIZE, NUM_OUTPUT_CLASSES, MAX_CLASS_ID
import numpy as np

from sklearn.metrics import confusion_matrix

CHECKPOINT_FOLDER = 'checkpoints'
CHECKPOINT_NAME = 'PVANET'
CHECKPOINT_STEP = 20


class Evaluate:
    def __init__(self):

        self.model = PVANet(evaluation=True,
                            batch_size=1,
                            image_dimensions=(None, None),
                            num_output_classes=MAX_CLASS_ID,
                            class_embeddings=None)

        # gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=.85)
        # self.sess = tf.Session(graph=self.model.graph,
        #                        config=tf.ConfigProto(gpu_options=gpu_options))

        self.sess = tf.Session(graph=self.model.graph)

    def evaluate(self):
        with self.model.graph.as_default():
            self.sess.run(tf.variables_initializer(tf.local_variables()))
            saver = tf.train.Saver()

            latest_checkpoint = tf.train.latest_checkpoint(CHECKPOINT_FOLDER)
            self.sess.run(tf.variables_initializer(tf.local_variables()))

            self.log("loading from checkpoint file: " + latest_checkpoint)
            saver.restore(self.sess, latest_checkpoint)

            coord = tf.train.Coordinator()
            threads = tf.train.start_queue_runners(sess=self.sess, coord=coord)
            predictions = np.array([])
            truths = np.array([])
            count = 0
            try:
                while not coord.should_stop():
                    prediction_ids, truth_ids, = self.sess.run(
                        [self.model.class_ids, self.model.truth])
                    predictions = np.append(predictions, prediction_ids.flatten())
                    truths = np.append(truths, truth_ids.flatten())
                    count += 1
                    self.log(count)

            except tf.errors.OutOfRangeError:
                self.log('Done validation -- epoch limit reached')
            finally:
                coord.request_stop()

            coord.join(threads)
            self.sess.close()
            np.savetxt("confusion_matrix.txt", confusion_matrix(truths, predictions), fmt="%d", delimiter=",")

    @staticmethod
    def log(message):
        logger.info(message)


if __name__ == '__main__':
    e = Evaluate()
    e.evaluate()
