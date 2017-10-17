import tensorflow as tf
import numpy as np

from internal_logger import logger
from model import ADEResNet

CHECKPOINT_FOLDER = 'checkpoints'
CHECKPOINT_NAME = 'ADE-ResNet'
CHECKPOINT_STEP = 1000
VALIDATION_STEP = 150

flags = tf.app.flags
FLAGS = flags.FLAGS

flags.DEFINE_integer('batch_size', 16, 'Size of each training batch')
flags.DEFINE_integer('image_size', 256, 'Size of each training batch')
flags.DEFINE_integer('num_classes', 151, 'Size of each training batch')


class Train:
    def __init__(self, image_size, batch_size, num_classes):
        self.image_size = image_size
        self.batch_size = batch_size
        self.num_classes = num_classes
        self.model = ADEResNet(batch_size=self.batch_size,
                               image_size=self.image_size,
                               num_output_classes=self.num_classes)
        config = tf.ConfigProto(allow_soft_placement=True)
        self.sess = tf.Session(graph=self.model.graph, config=config)

    def train(self):
        with self.model.graph.as_default():

            merged = tf.summary.merge_all()
            train_writer = tf.summary.FileWriter("summaries", t.model.graph)
            validation_writer = tf.summary.FileWriter("summaries/validation", t.model.graph)
            saver = tf.train.Saver(max_to_keep=1, keep_checkpoint_every_n_hours=3)
            latest_checkpoint = tf.train.latest_checkpoint(CHECKPOINT_FOLDER)

            self.log("initializing variables")
            self.sess.run(tf.variables_initializer(tf.global_variables()))
            self.sess.run(tf.variables_initializer(tf.local_variables()))
            self.log("initialized variables")

            if latest_checkpoint:
                self.log("loading from checkpoint file: " + latest_checkpoint)
                saver.restore(self.sess, latest_checkpoint)
            else:
                self.log("checkpoint not found")

            self.log("Creating coordinator and queue runners")
            coord = tf.train.Coordinator()
            threads = tf.train.start_queue_runners(sess=self.sess, coord=coord)
            self.log("beginning training")
            last_step = 0
            try:
                while not coord.should_stop():
                    self.log("batch")
                    m, ids, _, loss, step, = self.sess.run(
                        [merged, self.model.input_id,
                         self.model.training_step,
                         self.model.loss,
                         self.model.global_step
                         # self.model.all_losses_by_id
                         ])

                    self.log("loss = %.5f" % loss)

                    # if np.any(np.isnan(all_losses)):
                    #     self.log(str(np.where(np.isnan(all_losses) is True)))

                    train_writer.add_summary(m, step)

                    if step % CHECKPOINT_STEP == 0:
                        saver.save(self.sess, CHECKPOINT_FOLDER + '/' + CHECKPOINT_NAME, global_step=step)
                    last_step = step

                    if step % VALIDATION_STEP == 0:
                        m, ids, = self.sess.run(
                            [merged,
                             self.model.input_id],
                            feed_dict={self.model.is_training: False})
                        validation_writer.add_summary(m, step)
                        self.log("validation on ids: %s" % str(ids))

            except tf.errors.OutOfRangeError:
                self.log('Done training -- epoch limit reached')
            finally:
                if last_step:
                    saver.save(self.sess, CHECKPOINT_FOLDER + '/' + CHECKPOINT_NAME, global_step=last_step)
                coord.request_stop()

            coord.join(threads)
            self.sess.close()

    @staticmethod
    def log(message):
        logger.info(message)


if __name__ == '__main__':
    t = Train(image_size=(FLAGS.image_size, FLAGS.image_size),
              batch_size=FLAGS.batch_size,
              num_classes=FLAGS.num_classes)
    t.train()
