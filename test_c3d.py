

import time
from six.moves import xrange
import tensorflow as tf
import input_data
import c3d_model
import scipy.io as sio
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
  weight_initial = sio.loadmat('models/c3d_ucf101_tf.mat', squeeze_me=True)['weights']
  batch_size = 3
  max_steps = 25000
  MOVING_AVERAGE_DECAY = 0.9999
  train_list = 'list/train_ucf101.list'
  test_list = 'list/test_ucf101.list'
  model_save_dir = './models'
  model_filename = tf.train.latest_checkpoint(model_save_dir)
  with tf.variable_scope('var_name') as var_scope:
    weights = {
      'wc1': _variable_with_weight_decay('wc1', [3, 3, 3, 3, 64], wd, weight_initial[0]),
      'wc2': _variable_with_weight_decay('wc2', [3, 3, 3, 64, 128], wd, weight_initial[2]),
      'wc3a': _variable_with_weight_decay('wc3a', [3, 3, 3, 128, 256], wd, weight_initial[4]),
      'wc3b': _variable_with_weight_decay('wc3b', [3, 3, 3, 256, 256], wd, weight_initial[6]),
      'wc4a': _variable_with_weight_decay('wc4a', [3, 3, 3, 256, 512], wd, weight_initial[8]),
      'wc4b': _variable_with_weight_decay('wc4b', [3, 3, 3, 512, 512], wd, weight_initial[10]),
      'wc5a': _variable_with_weight_decay('wc5a', [3, 3, 3, 512, 512], wd, weight_initial[12]),
      'wc5b': _variable_with_weight_decay('wc5b', [3, 3, 3, 512, 512], wd, weight_initial[14]),
      'wd1': _variable_with_weight_decay('wd1', [8192, 4096], wd, weight_initial[16]),
      'wd2': _variable_with_weight_decay('wd2', [4096, 4096], wd, weight_initial[18]),
      'out': _variable_with_weight_decay('wout', [4096, c3d_model.NUM_CLASSES], wd, weight_initial[20])
    }
    biases = {
      'bc1': _variable_with_weight_decay('bc1', [64],wd, weight_initial[1]),
      'bc2': _variable_with_weight_decay('bc2', [128], wd,weight_initial[3]),
      'bc3a': _variable_with_weight_decay('bc3a', [256],wd, weight_initial[5]),
      'bc3b': _variable_with_weight_decay('bc3b', [256],wd, weight_initial[7]),
      'bc4a': _variable_with_weight_decay('bc4a', [512],wd, weight_initial[9]),
      'bc4b': _variable_with_weight_decay('bc4b', [512], wd,weight_initial[11]),
      'bc5a': _variable_with_weight_decay('bc5a', [512],wd, weight_initial[13]),
      'bc5b': _variable_with_weight_decay('bc5b', [512],wd, weight_initial[15]),
      'bd1': _variable_with_weight_decay('bd1', [4096], wd,weight_initial[17]),
      'bd2': _variable_with_weight_decay('bd2', [4096],wd, weight_initial[19]),
      'out': _variable_with_weight_decay('bout', [c3d_model.NUM_CLASSES],wd, weight_initial[21]),
    }
def tower_acc(logit, labels):
  correct_pred = tf.equal(tf.argmax(logit, 1), labels)
  accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))
  return accuracy
def placeholder_inputs(batch_size):
  images_placeholder = tf.placeholder(tf.float32, shape=(batch_size,
                                                         c3d_model.NUM_FRAMES_PER_CLIP,
                                                         c3d_model.CROP_SIZE,
                                                         c3d_model.CROP_SIZE,
                                                         c3d_model.CHANNELS))
  labels_placeholder = tf.placeholder(tf.int64, shape=(batch_size))
  return images_placeholder, labels_placeholder


def run_test():
  config = Config()
  test_lines = list(open(config.test_list,'r'))
  train_lines = list(open(config.train_list,'r'))
  num_test_videos = len(test_lines)
  print("Number of test videos={}".format(num_test_videos))

  # Get the sets of images and labels for training, validation, and
  images_placeholder, labels_placeholder = placeholder_inputs(config.batch_size)

  logit = c3d_model.inference_c3d(images_placeholder, 0.6, config.weight_initial)
  norm_score = tf.nn.softmax(logit)
  accuracy = tower_acc(logit, labels_placeholder)
  saver = tf.train.Saver()
  sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True))
  init = tf.global_variables_initializer()
  sess.run(init)
  # Create a saver for writing training checkpoints.
  #saver.restore(sess, config.model_filename)
  next_start_pos = 0
  all_steps = int((num_test_videos - 1) /config.batch_size + 1)
  res_acc = 0
  for step in xrange(all_steps):
    start_time = time.time()
    test_images, test_labels, next_start_pos, _ = \
            input_data.read_clip_and_label(
                    test_lines,
                    batch_size=config.batch_size,
                    start_pos=next_start_pos
                    )
    acc = sess.run(accuracy,feed_dict={images_placeholder:test_images,
                                      labels_placeholder:test_labels})
    print(acc)
    res_acc = res_acc+acc
  print(res_acc/all_steps)
  print("done")

def main(_):
  run_test()

if __name__ == '__main__':
  tf.app.run()
