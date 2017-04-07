import tensorflow as tf

from class_embeddings import class_embedding_lookup_table
from internal_logger import logger
from model import PVANet
from settings import IMAGE_HEIGHT, IMAGE_WIDTH, BATCH_SIZE, NUM_OUTPUT_CLASSES, MAX_CLASS_ID

CHECKPOINT_FOLDER = 'checkpoints'
CHECKPOINT_NAME = 'PVANET'
CHECKPOINT_STEP = 20


class Train:
    def __init__(self, image_dimensions=(256, 192), batch_size=50):
        self.image_dimensions = image_dimensions
        self.batch_size = batch_size
        self.model = PVANet(training=True,
                            batch_size=self.batch_size,
                            image_dimensions=image_dimensions,
                            num_output_classes=MAX_CLASS_ID,
                            class_embeddings=None)

        # gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=.5)
        # self.sess = tf.Session(graph=self.model.graph,
        #                        config=tf.ConfigProto(gpu_options=gpu_options, log_device_placement=True))

        self.sess = tf.Session(graph=self.model.graph)

    def train(self):
        with self.model.graph.as_default():
            self.sess.run(tf.variables_initializer(tf.local_variables()))
            merged = tf.summary.merge_all()
            train_writer = tf.summary.FileWriter("summaries", t.model.graph)
            saver = tf.train.Saver(max_to_keep=5, keep_checkpoint_every_n_hours=2)

            latest_checkpoint = tf.train.latest_checkpoint(CHECKPOINT_FOLDER)
            self.sess.run(tf.variables_initializer(tf.local_variables()))
            if latest_checkpoint:
                self.log("loading from checkpoint file: " + latest_checkpoint)
                saver.restore(self.sess, latest_checkpoint)
            else:
                self.log("checkpoint not found, initializing variables.")
                self.sess.run(tf.variables_initializer(tf.global_variables()))

            coord = tf.train.Coordinator()
            threads = tf.train.start_queue_runners(sess=self.sess, coord=coord)
            avg_loss = .0
            try:
                while not coord.should_stop():
                    self.log("batch")
                    m, _, loss, step, = self.sess.run(
                        [merged,
                         self.model.optimizer,
                         self.model.loss,
                         self.model.global_step])

                    train_writer.add_summary(m, step)
                    avg_loss += (loss / float(CHECKPOINT_STEP))
                    if step % CHECKPOINT_STEP == 0:
                        self.log("past %d runs avg_loss: %.2f" % \
                                 (CHECKPOINT_STEP, avg_loss))
                        self.log("saved checkpoint at step " + str(step))
                        avg_loss, avg_pixel_mse, avg_vgg16_mse = .0, .0, .0
                        saver.save(self.sess, CHECKPOINT_FOLDER + '/' + CHECKPOINT_NAME, global_step=step)
            except tf.errors.OutOfRangeError:
                self.log('Done training -- epoch limit reached')
            finally:
                coord.request_stop()

            coord.join(threads)
            self.sess.close()

    @staticmethod
    def log(message):
        logger.info(message)


if __name__ == '__main__':
    t = Train(image_dimensions=(IMAGE_HEIGHT, IMAGE_WIDTH), batch_size=BATCH_SIZE)
    t.train()
