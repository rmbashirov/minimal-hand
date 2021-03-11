import cv2
# import keyboard
import numpy as np
import open3d as o3d
import pygame
import os
import os.path as osp
import json
import time
from transforms3d.axangles import axangle2mat

import config
from capture import OpenCVCapture
from hand_mesh import HandMesh
from kinematics import mpii_to_mano
from utils import OneEuroFilter, imresize
from wrappers import ModelPipeline
from copy import deepcopy
from utils import *


class MyCapture:
  """
  OpenCV wrapper to read from webcam.
  """
  def __init__(self, fp, side='left'):
    """
    Init.
    """
    with open(fp, 'rb') as f:
      self.d = pickle.load(f)
    self.i = -1
    self.frame_indexes = sorted(self.d.keys())
    self.n = len(self.frame_indexes)
    self.side = side
    self.flip_side = 'right'

  def read(self):
    """
    Read one frame. Note this function might be blocked by the sensor.

    Returns
    -------
    np.ndarray
      Read frame. Might be `None` is the webcam fails to get on frame.
    """
    self.i += 1
    if self.i == self.n:
      return None
    frame_index = self.frame_indexes[self.i % self.n]
    frame = self.d[frame_index][self.side]
    if self.side == self.flip_side:
      frame = np.flip(frame, axis=1)
    return frame



def live_application(capture, output_dirpath):
  model = ModelPipeline()

  frame_index = 0
  mano_params = []
  measure_time = True
  while True:
    frame_large = capture.read()
    if frame_large is None:
      print(f'none frame {frame_index}')
      # if frame_index == 0:
      #   continue
      break
    # if frame_large.shape[0] > frame_large.shape[1]:
    #   margin = int((frame_large.shape[0] - frame_large.shape[1]) / 2)
    #   frame_large = frame_large[margin:-margin]
    # else:
    #   margin = int((frame_large.shape[1] - frame_large.shape[0]) / 2)
    #   frame_large = frame_large[:, margin:-margin]

    frame = imresize(frame_large, (128, 128))

    if measure_time:
      ends1 = []
      ends2 = []
      for i in range(1000):
        start = time.time()
        _, theta_mpii = model.process(frame)
        end1 = time.time()
        theta_mano = mpii_to_mano(theta_mpii)
        end2 = time.time()
        ends1.append(end1 - start)
        ends2.append(end2 - start)
      t1 = np.mean(ends1[10:])
      t2 = np.mean(ends2[10:])
      print(f't1: {t1 * 1000:.2f}ms, {1 / t1:.2f}hz')
      print(f't2: {t2 * 1000:.2f}ms, {1 / t2:.2f}hz')
      return
    else:
      _, theta_mpii = model.process(frame)
      theta_mano = mpii_to_mano(theta_mpii)

    mano_params.append(deepcopy(theta_mano.tolist()))

    osp.join(output_dirpath, "%06d.jpg" % frame_index)
    frame_index += 1
  with open(osp.join(output_dirpath, f'{capture.side}.pickle'), 'w') as f:
    json.dump(mano_params, f)


if __name__ == '__main__':
  fn_no_ext = '000012'
  input_fp = f'/home/renat/workdir/data/kinect_pose_fitting/kinect_hands/hand_crops_vis/{fn_no_ext}.pickle'
  output_dirpath = f'/home/renat/workdir/data/kinect_pose_fitting/kinect_hands/test_hand_models/minimal_hand/output_mano_params/{fn_no_ext}'
  os.makedirs(output_dirpath, exist_ok=True)
  for side in ['left', 'right']:
    live_application(MyCapture(input_fp, side=side), output_dirpath)
