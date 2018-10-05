import tensorflow as tf
NUM_CLASSES = 101# The UCF-101 dataset has 101 classes
# Images are cropped to (CROP_SIZE, CROP_SIZE)
CROP_SIZE = 112
CHANNELS = 3
# Number of frames per video clip
NUM_FRAMES_PER_CLIP = 16

def inference_c3d(_X, _dropout, weights,training = True):

  # Convolution Layer
  net = tf.layers.conv3d(inputs=_X, filters=64, kernel_size=3, padding='SAME', activation=tf.nn.relu,
                         kernel_initializer=tf.constant_initializer(weights[0]),
                         bias_initializer=tf.constant_initializer(weights[1]))
  net = tf.layers.max_pooling3d(inputs=net, pool_size=(1, 2, 2), strides=(1, 2, 2), padding='SAME')

  net = tf.layers.conv3d(inputs=net, filters=128, kernel_size=3, padding='SAME', activation=tf.nn.relu,
                         kernel_initializer=tf.constant_initializer(weights[2]),
                         bias_initializer=tf.constant_initializer(weights[3]))
  net = tf.layers.max_pooling3d(inputs=net, pool_size=2, strides=2, padding='SAME')

  net = tf.layers.conv3d(inputs=net, filters=256, kernel_size=3, padding='SAME', activation=tf.nn.relu,
                         kernel_initializer=tf.constant_initializer(weights[4]),
                         bias_initializer=tf.constant_initializer(weights[5]))
  net = tf.layers.conv3d(inputs=net, filters=256, kernel_size=3, padding='SAME', activation=tf.nn.relu,
                         kernel_initializer=tf.constant_initializer(weights[6]),
                         bias_initializer=tf.constant_initializer(weights[7]))
  net = tf.layers.max_pooling3d(inputs=net, pool_size=2, strides=2, padding='SAME')

  net = tf.layers.conv3d(inputs=net, filters=512, kernel_size=3, padding='SAME', activation=tf.nn.relu,
                         kernel_initializer=tf.constant_initializer(weights[8]),
                         bias_initializer=tf.constant_initializer(weights[9]))
  net = tf.layers.conv3d(inputs=net, filters=512, kernel_size=3, padding='SAME', activation=tf.nn.relu,
                         kernel_initializer=tf.constant_initializer(weights[10]),
                         bias_initializer=tf.constant_initializer(weights[11]))
  net = tf.layers.max_pooling3d(inputs=net, pool_size=2, strides=2, padding='SAME')
  net = tf.pad(net, [[0, 0], [1, 1], [1, 1], [1, 1], [0, 0]])

  net = tf.layers.conv3d(inputs=net, filters=512, kernel_size=3, activation=tf.nn.relu, padding='VALID',
                         kernel_initializer=tf.constant_initializer(weights[12]),
                         bias_initializer=tf.constant_initializer(weights[13]))
  net = tf.pad(net, [[0, 0], [1, 1], [1, 1], [1, 1], [0, 0]])
  net = tf.layers.conv3d(inputs=net, filters=512, kernel_size=3, activation=tf.nn.relu, padding='VALID',
                         kernel_initializer=tf.constant_initializer(weights[14]),
                         bias_initializer=tf.constant_initializer(weights[15]))
  net = tf.layers.max_pooling3d(inputs=net, pool_size=2, strides=2, padding='SAME')

  net = tf.layers.flatten(net)
  net = tf.layers.dense(inputs=net, units=4096, activation=tf.nn.relu,
                        kernel_initializer=tf.constant_initializer(weights[16]),
                        bias_initializer=tf.constant_initializer(weights[17]))
  net = tf.identity(net, name='fc1')
  net = tf.layers.dropout(inputs=net, rate=_dropout, training=training)

  net = tf.layers.dense(inputs=net, units=4096, activation=tf.nn.relu,
                        kernel_initializer=tf.constant_initializer(weights[18]),
                        bias_initializer=tf.constant_initializer(weights[19]))
  net = tf.identity(net, name='fc2')
  net = tf.layers.dropout(inputs=net, rate=_dropout, training=training)

  net = tf.layers.dense(inputs=net, units=NUM_CLASSES, activation=None,
                        kernel_initializer=tf.constant_initializer(weights[20]),
                        bias_initializer=tf.constant_initializer(weights[21]))
  net = tf.identity(net, name='logits')

  return net
