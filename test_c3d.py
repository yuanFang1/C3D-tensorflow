

import time
from six.moves import xrange
import tensorflow as tf
import input_data
import c3d_model
import numpy as np

def _variable_on_cpu(name, shape, initializer):
  with tf.device('/cpu:0'):
    var = tf.get_variable(name, shape, initializer=initializer)
  return var
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
  train_list = 'list/train_ucf101.list'
  test_list = 'list/test_ucf101.list'
  model_save_dir = './models'
  model_filename = "./models/c3d_ucf101_finetune_whole_iter_20000_TF.model"
  with tf.variable_scope('var_name') as var_scope:
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
  num_test_videos = len(list(open(config.test_list,'r')))
  print("Number of test videos={}".format(num_test_videos))

  # Get the sets of images and labels for training, validation, and
  images_placeholder, labels_placeholder = placeholder_inputs(config.batch_size)

  logit = c3d_model.inference_c3d(images_placeholder, 0.6, config.batch_size, config.weights, config.biases)
  norm_score = tf.nn.softmax(logit)
  accuracy = tower_acc(logit, labels_placeholder)
  saver = tf.train.Saver()
  sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True))
  init = tf.global_variables_initializer()
  sess.run(init)
  # Create a saver for writing training checkpoints.
  saver.restore(sess, config.model_filename)
  # And then after everything is built, start the training loop.
  write_file = open("predict_ret.txt", "w+")
  next_start_pos = 0
  all_steps = int((num_test_videos - 1) /config.batch_size + 1)
  for step in xrange(all_steps):
    # Fill a feed dictionary with the actual set of images and labels
    # for this particular training step.
    start_time = time.time()
    test_images, test_labels, next_start_pos, _, valid_len = \
            input_data.read_clip_and_label(
                    config.test_list,
                    config.batch_size,
                    start_pos=next_start_pos
                    )
    predict_score = norm_score.eval(
            session=sess,
            feed_dict={images_placeholder: test_images}
            )
    for i in range(0, valid_len):
      true_label = test_labels[i],
      top1_predicted_label = np.argmax(predict_score[i])
      # Write results: true label, class prob for true label, predicted label, class prob for predicted label
      write_file.write('{}, {}, {}, {}\n'.format(
              true_label[0],
              predict_score[i][true_label],
              top1_predicted_label,
              predict_score[i][top1_predicted_label]))
      print('{}, {}, {}, {}\n'.format(
              true_label[0],
              predict_score[i][true_label],
              top1_predicted_label,
              predict_score[i][top1_predicted_label]))
  write_file.close()
  print("done")

def main(_):
  run_test()

if __name__ == '__main__':
  tf.app.run()
