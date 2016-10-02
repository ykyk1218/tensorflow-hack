#!/usr/bin/env python
# -*- coding: utf-8 -*-
import sys
import os
import numpy
import time
from datetime import datetime
import tensorflow as tf

import model

IMAGE_DATA = ['woman', 1]

IMAGE_SIZE=112
INPUT_SIZE=96
DST_INPUT_SIZE=56
NUM_CLASS=5
NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN = 500

flags = tf.app.flags
FLAGS = flags.FLAGS
flags.DEFINE_string('train', 'tensorflow_image.csv', 'File name of train data')
flags.DEFINE_string('test', 'test.txt', 'File name of test data')
flags.DEFINE_string('train_dir', './tmp/', 'Directory to put the training data.')
flags.DEFINE_integer('max_steps', 100, 'Number of steps to run trainer.')
flags.DEFINE_integer('batch_size', 120, 'Batch size Must divide evenly into the dataset sizes.')
flags.DEFINE_float('learning_rate', 1e-4, 'Initial learning rate.')

def load_data(csv, batch_size, shuffle, distored):
  queue = tf.train.string_input_producer(csv, shuffle=shuffle)
  reader = tf.TextLineReader()
  key, value = reader.read(queue)
  filename, label = tf.decode_csv(value, [["path"], [1]])
  

  label = tf.cast(label, tf.int64)
  label = tf.one_hot(label, depth = 5, on_value=1.0, off_value=0.0, axis=-1)
  
  jpeg_r = tf.read_file('./images/' + filename)
  image = tf.image.decode_jpeg(jpeg_r, channels=3)
  image = tf.cast(image, tf.float32)
  image.set_shape([IMAGE_SIZE, IMAGE_SIZE, 3])

  #if distored:

  image = tf.image.resize_images(image, DST_INPUT_SIZE, DST_INPUT_SIZE)
  image = tf.image.per_image_whitening(image)

  # Ensure that the random shuffling has good mixing properties.<Paste>
  min_fraction_of_examples_in_queue = 0.4
  min_queue_examples = int(NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN * min_fraction_of_examples_in_queue)

  return _generate_image_and_label_batch(
         image,
         label,
         filename,
         min_queue_examples, batch_size,
         shuffle=shuffle)

def _generate_image_and_label_batch(image, label, filename, min_queue_examples,
                                    batch_size, shuffle):
  num_preprocess_threads = 16
  capacity = min_queue_examples + 3 * batch_size

  if shuffle:
    images, label_batch, filename = tf.train.shuffle_batch(
        [image, label, filename],
        batch_size=batch_size,
        num_threads=num_preprocess_threads,
        capacity=capacity,
        min_after_dequeue=min_queue_examples)
  else:
    images, label_batch, filename = tf.train.batch(
        [image, label, filename],
        batch_size=batch_size,
        num_threads=num_preprocess_threads,
        capacity=min_queue_examples + 3 * batch_size)

  tf.image_summary('image', images, max_images=100)
  labels = tf.reshape(label_batch, [batch_size, NUM_CLASS])
  return images, labels, filename


def main(ckpt = None):
  with tf.Graph().as_default():
    keep_prob = tf.placeholder("float")

    images, labels, filename = load_data([FLAGS.train], FLAGS.batch_size, shuffle = True, distored = True)
    logits = model.inference(images, keep_prob, DST_INPUT_SIZE, NUM_CLASS)
    loss_value = model.loss(logits, labels)
    train_op = model.training(loss_value, FLAGS.learning_rate)
    acc = model.accuracy(logits, labels)

    saver = tf.train.Saver(max_to_keep = 0)
    sess = tf.Session()
    init = tf.initialize_all_variables()
    sess.run(init)
    if ckpt:
      print('restore ckpt', ckpt)
      saver.restore(sess, ckpt)
    tf.train.start_queue_runners(sess)

    summary_op = tf.merge_all_summaries()
    summary_writer = tf.train.SummaryWriter(FLAGS.train_dir, sess.graph)

    for step in range(FLAGS.max_steps):
        print(step)
        start_time = time.time()
        _, loss_result, acc_res = sess.run([train_op, loss_value, acc], feed_dict={keep_prob: 0.99})
        duration = time.time() - start_time

        if step % 10 == 0:
            num_examples_per_step = FLAGS.batch_size
            examples_per_sec = num_examples_per_step / duration
            sec_per_batch = float(duration)
            format_str = ('%s: step %d, loss = %.2f (%.1f examples/sec; %.3f sec/batch)')
            print (format_str % (datetime.now(), step, loss_result, examples_per_sec, sec_per_batch))

        if step % 100 == 0:
            summary_str = sess.run(summary_op, feed_dict={keep_prob: 1.0})
            summary_writer.add_summary(summary_str, step)

        if step % 1000 == 0 or (step + 1) == FLAGS.max_steps or loss_result ==0:
            checkpoint_path = os.path.join(FLAGS.train_dir, 'model.ckpt')
            saver.save(sess, checkpoint_path, global_step=step)

        if loss_result == 0:
            print("loss is zero")
            break
if __name__ == '__main__':
  ckpt = None
  if len(sys.argv) == 2:
    ckpt = sys.argv[1]
  main(ckpt)
