import numpy as np
from transforms3d.quaternions import quat2mat

from config import *
from kinematics import *
from utils import *


class HandMesh():
  """
  Wrapper for the MANO hand model.
  """
  def __init__(self, model_path):
    """
    Init.

    Parameters
    ----------
    model_path : str
      Path to the MANO model file. This model is converted by `prepare_mano.py`
      from official release.
    """
    params = load_pkl(model_path)
    self.verts = params['verts']
    print('verts', self.verts.shape)
    self.faces = params['faces']
    self.weights = params['weights']
    print('weights', self.weights.shape)
    self.joints = params['joints']

    self.n_verts = self.verts.shape[0]
    self.n_faces = self.faces.shape[0]

    self.ref_pose = []
    self.ref_T = []
    for j in range(MANOHandJoints.n_joints):
      parent = MANOHandJoints.parents[j]
      if parent is None:
        self.ref_T.append(self.verts)
        self.ref_pose.append(self.joints[j])
      else:
        self.ref_T.append(self.verts - self.joints[parent])
        self.ref_pose.append(self.joints[j] - self.joints[parent])
    self.ref_pose = np.expand_dims(np.stack(self.ref_pose, 0), -1)
    self.ref_T = np.expand_dims(np.stack(self.ref_T, 1), -1)
    print('ref_T', self.ref_T.shape)

  def set_abs_quat(self, quat):
    """
    Set absolute (global) rotation for the hand.

    Parameters
    ----------
    quat : np.ndarray, shape [J, 4]
      Absolute rotations for each joint in quaternion.

    Returns
    -------
    np.ndarray, shape [V, 3]
      Mesh vertices after posing.
    """
    mats = []
    for j in range(MANOHandJoints.n_joints):
      # mats.append(np.eye(3))
      mat = quat2mat(quat[j])
      print(j, mat)
      mats.append(mat)
      # mats.append(quat2mat(quat[0]))
      # print(quat2mat(quat[0]))
      # if j == 0:
      #   print(mats[-1])
    mats = np.stack(mats, 0)

    pose = np.matmul(mats, self.ref_pose)
    joint_xyz = [None] * MANOHandJoints.n_joints
    for j in range(MANOHandJoints.n_joints):
      joint_xyz[j] = pose[j]
      parent = MANOHandJoints.parents[j]
      if parent is not None:
        joint_xyz[j] += joint_xyz[parent]
    joint_xyz = np.stack(joint_xyz, 0)[..., 0]
    np.save('/home/renat/Desktop/joint_zyx.npy', joint_xyz)

    T = np.matmul(np.expand_dims(mats, 0), self.ref_T)[..., 0]
    self.verts = [None] * MANOHandJoints.n_joints
    for j in range(MANOHandJoints.n_joints):
      self.verts[j] = T[:, j]
      parent = MANOHandJoints.parents[j]
      if parent is not None:
        self.verts[j] += joint_xyz[parent]
    self.verts = np.stack(self.verts, 1)
    self.verts = np.sum(self.verts * self.weights, 1)
    np.save('/home/renat/Desktop/inf_verts.npy', self.verts)

    return self.verts.copy()
