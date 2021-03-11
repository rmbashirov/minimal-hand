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
  """
  Launch an application that reads from a webcam and estimates hand pose at
  real-time.

  The captured hand must be the right hand, but will be flipped internally
  and rendered.

  Parameters
  ----------
  capture : object
    An object from `capture.py` to read capture stream from.
  """
  ############ output visualization ############
  view_mat = axangle2mat([1, 0, 0], np.pi) # align different coordinate systems
  window_size = 1080

  hand_mesh = HandMesh(config.HAND_MESH_MODEL_PATH)
  mesh = o3d.geometry.TriangleMesh()
  mesh.triangles = o3d.utility.Vector3iVector(hand_mesh.faces)
  mesh.vertices = \
    o3d.utility.Vector3dVector(np.matmul(view_mat, hand_mesh.verts.T).T * 1000)
  mesh.compute_vertex_normals()

  viewer = o3d.visualization.Visualizer()
  viewer.create_window(
    width=window_size + 1, height=window_size + 1,
    window_name='Minimal Hand - output'
  )
  viewer.add_geometry(mesh)

  view_control = viewer.get_view_control()
  cam_params = view_control.convert_to_pinhole_camera_parameters()
  extrinsic = cam_params.extrinsic.copy()
  extrinsic[0:3, 3] = 0
  cam_params.extrinsic = extrinsic
  cam_params.intrinsic.set_intrinsics(
    window_size + 1, window_size + 1, config.CAM_FX, config.CAM_FY,
    window_size // 2, window_size // 2
  )
  view_control.convert_from_pinhole_camera_parameters(cam_params)
  view_control.set_constant_z_far(1000)

  render_option = viewer.get_render_option()
  render_option.load_from_json('./render_option.json')
  viewer.update_renderer()

  ############ input visualization ############
  pygame.init()
  display = pygame.display.set_mode((window_size, window_size))
  pygame.display.set_caption('Minimal Hand - input')

  ############ misc ############
  mesh_smoother = OneEuroFilter(4.0, 0.0)
  clock = pygame.time.Clock()
  model = ModelPipeline()

  frame_index = 0
  mano_params = []
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

    _, theta_mpii = model.process(frame)
    theta_mano = mpii_to_mano(theta_mpii)

    mano_params.append(deepcopy(theta_mano.tolist()))

    osp.join(output_dirpath, "%06d.jpg" % frame_index)

    v = hand_mesh.set_abs_quat(theta_mano)
    v *= 2 # for better visualization
    v = v * 1000 + np.array([0, 0, 400])
    v = mesh_smoother.process(v)
    mesh.triangles = o3d.utility.Vector3iVector(hand_mesh.faces)
    mesh.vertices = o3d.utility.Vector3dVector(np.matmul(view_mat, v.T).T)
    mesh.paint_uniform_color(config.HAND_COLOR)
    mesh.compute_triangle_normals()
    mesh.compute_vertex_normals()
    viewer.update_geometry()
    viewer.poll_events()
    viewer.capture_screen_image(osp.join(output_dirpath, "%06d.jpg" % frame_index))

    display.blit(
      pygame.surfarray.make_surface(
        np.transpose(
          imresize(frame_large, (window_size, window_size)
        ), (1, 0, 2))
      ),
      (0, 0)
    )
    pygame.display.update()

    # if keyboard.is_pressed("esc"):
    #   break

    clock.tick(30)
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
