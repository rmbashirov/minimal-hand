import os

# if not os.path.exists('/Vol1'):
#     DETECTION_MODEL_PATH = '/home/renat/workdir/data/kinect_pose_fitting/kinect_hands/test_hand_models/minimal_hand/model/detnet/detnet.ckpt'
#     IK_MODEL_PATH = '/home/renat/workdir/data/kinect_pose_fitting/kinect_hands/test_hand_models/minimal_hand/model/iknet/iknet.ckpt'
#     HAND_MESH_MODEL_PATH = '/home/renat/workdir/data/kinect_pose_fitting/kinect_hands/test_hand_models/minimal_hand/model/hand_mesh/hand_mesh_model.pkl'
# else:
#     DETECTION_MODEL_PATH = '/Vol0/user/r.bashirov/workdir/git/data/smplx_kinect/minimal-hand/model/detnet/detnet.ckpt'
#     IK_MODEL_PATH = '/Vol0/user/r.bashirov/workdir/git/data/smplx_kinect/minimal-hand/model/iknet/iknet.ckpt'
#     HAND_MESH_MODEL_PATH = '/Vol0/user/r.bashirov/workdir/git/data/smplx_kinect/minimal-hand/model/hand_mesh/hand_mesh_model.pkl'


# use left hand
# OFFICIAL_MANO_PATH = '/mnt/hdd10/mano/mano_v1_2/models/MANO_LEFT.pkl'
IK_UNIT_LENGTH = 0.09473151311686484  # in meter

HAND_COLOR = [228/255, 178/255, 148/255]

# only for rendering
CAM_FX = 620.744
CAM_FY = 621.151
