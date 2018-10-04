import os
import time
import numpy
from six.moves import xrange  # pylint: disable=redefined-builtin
import tensorflow as tf
import input_data
import c3d_model
import math
import numpy as np

def _variable_with_weight_decay(name, shape, wd):
  var = _variable_on_cpu(name, shape, tf.contrib.layers.xavier_initializer())
  if wd is not None:
    weight_decay = tf.nn.l2_loss(var)*wd
    tf.add_to_collection('weightdecay_losses', weight_decay)
  return var

class Config(object):
    dropout = 0.5
    weight_initial = 0.0005
    bias_initial = 0.0
    batch_size = 10
    max_steps = 5000
    MOVING_AVERAGE_DECAY = 0.9999
    train_list = 'list/train_ucf11.list'
    test_list = 'list/test_ucf11.list'
    model_save_dir = './models'
    model_filename = tf.train.latest_checkpoint(model_save_dir)
    weights = {
        'wc1': _variable_with_weight_decay('wc1', [3, 3, 3, 3, 64], weight_initial),
        'wc2': _variable_with_weight_decay('wc2', [3, 3, 3, 64, 128], weight_initial),
        'wc3a': _variable_with_weight_decay('wc3a', [3, 3, 3, 128, 256], weight_initial),
        'wc3b': _variable_with_weight_decay('wc3b', [3, 3, 3, 256, 256], weight_initial),
        'wc4a': _variable_with_weight_decay('wc4a', [3, 3, 3, 256, 512], weight_initial),
        'wc4b': _variable_with_weight_decay('wc4b', [3, 3, 3, 512, 512], weight_initial),
        'wc5a': _variable_with_weight_decay('wc5a', [3, 3, 3, 512, 512], weight_initial),
        'wc5b': _variable_with_weight_decay('wc5b', [3, 3, 3, 512, 512], weight_initial),
        'wd1': _variable_with_weight_decay('wd1', [8192, 4096], weight_initial),
        'wd2': _variable_with_weight_decay('wd2', [4096, 4096], weight_initial),
        'out': _variable_with_weight_decay('wout', [4096, c3d_model.NUM_CLASSES], weight_initial)
    }
    biases = {
        'bc1': _variable_with_weight_decay('bc1', [64], bias_initial),
        'bc2': _variable_with_weight_decay('bc2', [128], bias_initial),
        'bc3a': _variable_with_weight_decay('bc3a', [256], bias_initial),
        'bc3b': _variable_with_weight_decay('bc3b', [256], bias_initial),
        'bc4a': _variable_with_weight_decay('bc4a', [512], bias_initial),
        'bc4b': _variable_with_weight_decay('bc4b', [512], bias_initial),
        'bc5a': _variable_with_weight_decay('bc5a', [512], bias_initial),
        'bc5b': _variable_with_weight_decay('bc5b', [512], bias_initial),
        'bd1': _variable_with_weight_decay('bd1', [4096], bias_initial),
        'bd2': _variable_with_weight_decay('bd2', [4096], bias_initial),
        'out': _variable_with_weight_decay('bout', [c3d_model.NUM_CLASSES], bias_initial),
    }

def placeholder_inputs(batch_size):
  images_placeholder = tf.placeholder(tf.float32, shape=(batch_size,
                                                         c3d_model.NUM_FRAMES_PER_CLIP,
                                                         c3d_model.CROP_SIZE,
                                                         c3d_model.CROP_SIZE,
                                                         c3d_model.CHANNELS))
  labels_placeholder = tf.placeholder(tf.int64, shape=(batch_size))
  return images_placeholder, labels_placeholder

def tower_loss(name_scope, logit, labels):
  cross_entropy_mean = tf.reduce_mean(
                  tf.nn.sparse_softmax_cross_entropy_with_logits(labels=labels,logits=logit)
                  )
  tf.summary.scalar(
                  name_scope + '_cross_entropy',
                  cross_entropy_mean
                  )
  weight_decay_loss = tf.get_collection('weightdecay_losses')
  tf.summary.scalar(name_scope + '_weight_decay_loss', tf.reduce_mean(weight_decay_loss) )

  # Calculate the total loss for the current tower.
  total_loss = cross_entropy_mean + weight_decay_loss 
  tf.summary.scalar(name_scope + '_total_loss', tf.reduce_mean(total_loss) )
  return total_loss

def tower_acc(logit, labels):
  correct_pred = tf.equal(tf.argmax(logit, 1), labels)
  accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))
  return accuracy

def _variable_on_cpu(name, shape, initializer):
  with tf.device('/cpu:0'):
    var = tf.get_variable(name, shape, initializer=initializer)
  return var

def run_training():
  config = Config()
  if not os.path.exists(config.model_save_dir):
      os.makedirs(config.model_save_dir)
  use_pretrained_model = True 

  with tf.Graph().as_default():
    global_step = tf.Variable(0, name='global_step', trainable=False,dtype=tf.float32)
    images_placeholder, labels_placeholder = placeholder_inputs(config.batch_size)

    logit = c3d_model.inference_c3d(
                    images_placeholder,
                    0.5,
                    config.batch_size,
                    config.weights,
                    config.biases
                    )
    loss_name_scope = ('loss')
    loss = tower_loss(
                    loss_name_scope,
                    logit,
                    labels_placeholder
                    )
    train_op = tf.train.AdamOptimizer(1e-3).minimize(loss,global_step=global_step)
    accuracy = tower_acc(logit, labels_placeholder)
    tf.summary.scalar('accuracy', accuracy)

    # Create a saver for writing training checkpoints.
    saver = tf.train.Saver()
    init = tf.global_variables_initializer()

    # Create a session for running Ops on the Graph.
    with tf.Session(config=tf.ConfigProto(allow_soft_placement=True)) as sess:
      sess.run(init)
      if config.model_filename!=None  and use_pretrained_model:
        saver.restore(sess, config.model_filename)

      # Create summary writter
      merged = tf.summary.merge_all()
      train_writer = tf.summary.FileWriter('./visual_logs/train', sess.graph)
      test_writer = tf.summary.FileWriter('./visual_logs/test', sess.graph)
      minstep = np.int(sess.run(global_step))
      print(minstep)
      for step in xrange(minstep,config.max_steps):
        global_step=global_step+1
        start_time = time.time()
        train_images, train_labels, _, _, _ = input_data.read_clip_and_label(
                        filename=config.train_list,
                        batch_size=config.batch_size ,
                        num_frames_per_clip=c3d_model.NUM_FRAMES_PER_CLIP,
                        crop_size=c3d_model.CROP_SIZE,
                        shuffle=True
                        )
        sess.run(train_op, feed_dict={
                        images_placeholder: train_images,
                        labels_placeholder: train_labels
                        })
        duration = time.time() - start_time
        print('Step %d: %.3f sec' % (step, duration))

        # Save a checkpoint and evaluate the model periodically.
        if (step) % 10 == 0 or (step + 1) == config.max_steps:
          saver.save(sess, os.path.join(config.model_save_dir, 'c3d_ucf_model'), global_step=step)
          print('Training Data Eval:')
          summary, acc = sess.run(
                          [merged, accuracy],
                          feed_dict={images_placeholder: train_images,
                              labels_placeholder: train_labels
                              })
          print ("accuracy: " + "{:.5f}".format(acc))
          train_writer.add_summary(summary, step)
          print('Validation Data Eval:')
          val_images, val_labels, _, _, _ = input_data.read_clip_and_label(
                          filename=config.test_list,
                          batch_size=config.batch_size,
                          num_frames_per_clip=c3d_model.NUM_FRAMES_PER_CLIP,
                          crop_size=c3d_model.CROP_SIZE,
                          shuffle=True
                          )
          summary, acc = sess.run(
                          [merged, accuracy],
                          feed_dict={images_placeholder: val_images,
                                     labels_placeholder: val_labels})
          print ("accuracy: " + "{:.5f}".format(acc))
          test_writer.add_summary(summary, step)
  print("done")

def main(_):
  run_training()

if __name__ == '__main__':
  tf.app.run()
