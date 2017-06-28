import tensorflow as tf
import preprocess_utility as ult
from imagenet_data import ImagenetData

tf.app.flags.DEFINE_integer('batch_size', 32,
                            """Number of images to process in a batch.""")
tf.app.flags.DEFINE_integer('image_size', 224,
                            """Provide square images of this size.""")
tf.app.flags.DEFINE_integer('num_preprocess_threads', 4,
                            """Number of preprocessing threads per tower. """
                            """Please make this a multiple of 4.""")
tf.app.flags.DEFINE_integer('num_readers', 4,
                            """Number of parallel readers during train.""")
IMAGE_SIZE=FLAGS.image_size

FLAGS = tf.app.flags.FLAGS
tf.app.flags.DEFINE_string('subset', 'train',
"""Either 'train' or 'validation'.""")

# def main():
#     dataset = ImagenetData(subset=FLAGS.subset)
#     data_files = dataset.data_files()
#     images, labels = ult.distorted_inputs(
#         dataset,
#         batch_size=FLAGS.batch_size,
#         num_preprocess_threads=FLAGS.num_preprocess_threads)
#
def traverse_train():
    dataset = ImagenetData(subset=FLAGS.subset)
    data_files = dataset.data_files()

    if data_files is None:
        raise ValueError('No data files found for this dataset')
    else:
        print(data_files)
#
#     # Create filename_queue
#     if train:
#         filename_queue = tf.train.string_input_producer(data_files,
#         shuffle=False,
#         capacity=16)
#     else:
#         filename_queue = tf.train.string_input_producer(data_files,
#         shuffle=False,
#         capacity=1)
#
#     if num_preprocess_threads is None:
#         num_preprocess_threads =  4
#         num_readers = 1
#
#     if num_readers > 1:
#         enqueue_ops = []
#         for _ in range(num_readers):
#             reader = dataset.reader()
#             _, value = reader.read(filename_queue)
#             enqueue_ops.append(examples_queue.enqueue([value]))
#
#             tf.train.queue_runner.add_queue_runner(
#             tf.train.queue_runner.QueueRunner(examples_queue, enqueue_ops))
#             example_serialized = examples_queue.dequeue()
#     else:
#         reader = dataset.reader()
#         key, example_serialized = reader.read(filename_queue)
#         print(key)
#
#     images_and_labels = []
#     for thread_id in range(num_preprocess_threads):
#         # Parse a serialized Example proto to extract the image and metadata.
#         image_buffer, label_index, bbox, _ = parse_example_proto(
#         example_serialized)
#         image = image_preprocessing(image_buffer, bbox, train, thread_id)
#         images_and_labels.append([image, label_index])
def main(argv=None):  # pylint: disable=unused-argument
  traverse_train()

if __name__ == '__main__':
  tf.app.run()
