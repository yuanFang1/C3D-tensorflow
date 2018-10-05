from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
from six.moves import xrange  # pylint: disable=redefined-builtin
import tensorflow as tf
import PIL.Image as Image
import random
import numpy as np
from scipy.misc import imread, imresize, imsave
import time

def get_frames_data(paths, offsets, size=(128, 171), crop_size=(112, 112), mode='RGB', interp='bilinear'):
  assert mode in ('RGB', 'L'), 'Mode is either RGB or L'

  clips = []
  for file_name in paths:
    # Read video frame
    im = imread(file_name, mode=mode)

    # Resize frame to init resolution and crop then resize to target resolution
    if mode == 'RGB':
      im = imresize(im, size=size, interp=interp)
      data = im[offsets[0]:offsets[1], offsets[2]:offsets[3], :]
      im = imresize(data, size=crop_size, interp=interp)
    else:
      im = imresize(im, size=size, interp=interp)
      data = im[offsets[0]:offsets[1], offsets[2]:offsets[3]]
      im = imresize(data, size=crop_size, interp=interp)

    clips.append(im)

  clips = np.array(clips, dtype=np.float32)

  if mode == 'RGB':
    return clips
  return np.expand_dims(clips, axis=3)

def read_clip_and_label(input_lines, batch_size, start_pos=-1, num_frames_per_clip=16, crop_size=112, shuffle=False):
  read_dirnames = []
  data = []
  label = []
  batch_index = 0
  next_batch_start = -1
  lines = input_lines
  if start_pos < 0:
    shuffle = True
  if shuffle:
    video_indices = list(range(len(lines)))
    random.seed(time.time())
    random.shuffle(video_indices)
  else:
    # Process videos sequentially
    video_indices = range(start_pos, len(lines))
  for index in video_indices:
    if(batch_index>=batch_size):
      next_batch_start = index
      break
    line = lines[index].strip('\n').split()
    dirname = line[0]
    tmp_label = line[1]
    if not shuffle:
      print("Loading a video clip from {}...".format(dirname))
    v_paths = [dirname + '/%05d.jpg' % (f + 1) for f in range(0, 0 + num_frames_per_clip)]
    offsets = [8, 8 + 112, 30, 30 + 112]  # center crop
    img_datas= get_frames_data(v_paths,offsets)
    if(len(img_datas)!=0):
      data.append(img_datas)
      label.append(int(tmp_label))
      batch_index = batch_index + 1
      read_dirnames.append(dirname)


  np_arr_data = np.array(data).astype(np.float32)
  np_arr_label = np.array(label).astype(np.int64)

  return np_arr_data, np_arr_label, next_batch_start, read_dirnames
