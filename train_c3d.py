import os
import time
import scipy.io as sio
from six.moves import xrange  # pylint: disable=redefined-builtin
import tensorflow as tf
import input_data
import c3d_model
import math
import numpy as np
def _variable_on_cpu(name, shape, initializer):
  with tf.device('/cpu:0'):
    var = tf.get_variable(name, shape, initializer=tf.constant_initializer(initializer))
  return var
def _variable_with_weight_decay(name, shape, wd,initializer):
  var = _variable_on_cpu(name, shape, initializer)
  if wd is not None:
    weight_decay = tf.nn.l2_loss(var)*wd
    tf.add_to_collection('weightdecay_losses', weight_decay)
  return var

class Config(object):
    dropout = 0.5
    wd = 0.0005
    weight_initial =  sio.loadmat('models/c3d_ucf101_tf.mat', squeeze_me=True)['weights']
    batch_size = 3
    max_steps = 25000
    MOVING_AVERAGE_DECAY = 0.9999
    train_list = 'list/train_ucf101.list'
    test_list = 'list/test_ucf101.list'
    use_pretrained_model = False
    model_save_dir = './models'
    model_filename = tf.train.latest_checkpoint(model_save_dir)

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
  weight_decay_loss = tf.get_collection('weightdecay_losses')
  total_loss = cross_entropy_mean + weight_decay_loss 
  return total_loss

def tower_acc(logit, labels):
  correct_pred = tf.equal(tf.argmax(logit, 1), labels)
  accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))
  return accuracy

def run_training():
  config = Config()
  test_lines = open(config.test_list, 'r')
  train_lines = open(config.train_list, 'r')
  if not os.path.exists(config.model_save_dir):
      os.makedirs(config.model_save_dir)
  global_step = tf.Variable(0, name='global_step', trainable=False,dtype=tf.float32)
  images_placeholder, labels_placeholder = placeholder_inputs(config.batch_size)

  logit = c3d_model.inference_c3d(
                  images_placeholder,
                  0.5,
                  config.weight_initial,
                  )
  loss_name_scope = ('loss')
  loss = tower_loss(
                  loss_name_scope,
                  logit,
                  labels_placeholder
                  )
  train_op = tf.train.AdamOptimizer(1e-3).minimize(loss,global_step=global_step)
  accuracy = tower_acc(logit, labels_placeholder)

  # Create a saver for writing training checkpoints.
  saver = tf.train.Saver()
  init = tf.global_variables_initializer()

  # Create a session for running Ops on the Graph.
  with tf.Session(config=tf.ConfigProto(allow_soft_placement=True)) as sess:
    sess.run(init)
    if config.model_filename!=None  and config.use_pretrained_model:
      saver.restore(sess, config.model_filename)

    minstep = np.int(sess.run(global_step))
    print(minstep)
    for step in xrange(minstep,config.max_steps):
      global_step=global_step+1
      start_time = time.time()
      train_images, train_labels, _, _= input_data.read_clip_and_label(
                      input_lines=train_lines,
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
        acc = sess.run([accuracy],
                        feed_dict={images_placeholder: train_images,
                            labels_placeholder: train_labels
                            })
        print ("accuracy: " + "{:.5f}".format(acc))
        print('Validation Data Eval:')
        val_images, val_labels, _, _ = input_data.read_clip_and_label(
                        input_lines=test_lines,
                        batch_size=config.batch_size,
                        num_frames_per_clip=c3d_model.NUM_FRAMES_PER_CLIP,
                        crop_size=c3d_model.CROP_SIZE,
                        shuffle=True
                        )
        acc = sess.run( [accuracy],
                        feed_dict={images_placeholder: val_images,
                                   labels_placeholder: val_labels})
        print ("accuracy: " + "{:.5f}".format(acc))
  print("done")

def main(_):
  run_training()

if __name__ == '__main__':
  tf.app.run()
