import os
import os.path as osp
import numpy as np
import argparse
import pickle
from collections import defaultdict
import time
from copy import deepcopy

from PIL import Image
import cv2
from skimage import io
from azure_people_unpacked import AzurePeopleUnpackedDataset


from utils import imresize
from wrappers import ModelPipeline
from kinematics import mpii_to_mano


def pil_crop(image, bbox):
    """Crops area from image specified as bbox. Always returns area of size as bbox filling$
    Args:
        image numpy array of shape (height, width, 3): input image
        bbox tuple of size 4: input bbox (left, upper, right, lower)
    Returns:
        cropped_image numpy array of shape (height, width, 3): resulting cropped image
    """
    image_pil = Image.fromarray(image, mode='RGB')
    image_pil = image_pil.crop(bbox)
    return np.asarray(image_pil)


def get_cube(p, s=0.15):
    diffs = np.array([
        [-1, -1, -1, -1, 1, 1, 1, 1],
        [-1, -1, 1, 1, -1, -1, 1, 1],
        [-1, 1, -1, 1, -1, 1, -1, 1]
    ], dtype=np.float32) * s

    return (diffs + p.reshape(3, 1)).T


def depth3d_to_color2d(points, cam_params):
    points = points @ cv2.Rodrigues(cam_params.depth2rgb_rvec)[0].T + cam_params.depth2rgb_tvec.reshape(1, -1)
    points_2d = points @ cam_params.K_rgb_undist.T
    points_2d = points_2d[:, :2] / points_2d[:, [2]]
    return points_2d


def get_bbox(points):
    left = np.min(points[:, 0])
    right = np.max(points[:, 0])
    top = np.min(points[:, 1])
    bottom = np.max(points[:, 1])
    h = bottom - top
    w = right - left
    if h > w:
        cx = (left + right) / 2
        left = cx - h / 2
        right = left + h
    else:
        cy = (bottom + top) / 2
        top = cy - w / 2
        bottom = top + w
    return left, top, right, bottom


class Processor:
    def __init__(self):
        self.model = ModelPipeline()
        self.flip_side = 'right'

    def process(self, img, side):
        frame = imresize(img, (128, 128))

        if side == self.flip_side:
            frame = np.flip(frame, axis=1)

        _, theta_mpii = self.model.process(frame)
        theta_mano = mpii_to_mano(theta_mpii)
        return theta_mano


def process_pid(pid, sn):
    print(f'{"=" * 20} pid: {pid}, sn: {sn} {"=" * 20}')

    # kinect_unpacked_dp = "/Vol1/dbstore/datasets/violet/AzurePeople/offline_processor2/align_06"
    # kinect_unpacked_dp_bt = "/Vol1/dbstore/datasets/violet/AzurePeople/offline_processor2/align_02"
    # output_dirpath = "/Vol0/user/r.bashirov/workdir/git/data/smplx_kinect/inf/AzurePeople/minimal_hand"

    kinect_unpacked_dp = "/Vol0/user/r.bashirov/workdir/git/data/smplx_kinect/test_capture/offline_processor2"
    kinect_unpacked_dp_bt = kinect_unpacked_dp
    output_dirpath = "/Vol0/user/r.bashirov/workdir/git/data/smplx_kinect/inf/AzureTest/minimal_hand"

    output_dirpath = osp.join(output_dirpath, sn)
    os.makedirs(output_dirpath, exist_ok=True)

    apud = AzurePeopleUnpackedDataset(
        osp.join(kinect_unpacked_dp, pid),
        serial_numbers=[sn]
    )
    apud.dataset_dirpath = osp.join(kinect_unpacked_dp_bt, pid)
    apud.read_bt()
    apud.dataset_dirpath = osp.join(kinect_unpacked_dp, pid)
    apud.parse_cam_params()

    color_dirpath = osp.join(apud.dataset_dirpath, sn, 'color_undistorted')
    fns_dict = dict()
    for fn in os.listdir(color_dirpath):
        fn_no_ext = osp.splitext(fn)[0]
        try:
            frame_index = int(fn_no_ext)
            fns_dict[frame_index] = fn
        except:
            pass

    processor = Processor()
    result = defaultdict(dict)
    done = 0

    start = time.time()
    # indexes = np.array(list(sorted(fns_dict.keys())))
    # indexes = indexes[np.logical_and(indexes > 3400, indexes < 3500)]
    # print(indexes)
    # return
    for frame_index in sorted(fns_dict.keys()):
        img_path = osp.join(color_dirpath, fns_dict[frame_index])
        color_undistorted = io.imread(img_path)[:, :, :3]
        try:
            kinect_joints = apud.get_bt_joints(frame_index, sn)
            kinect_joints = np.array(kinect_joints) / 1000
        except:
            print(f'no joints for frame {frame_index}')
            continue
        for hand_index, side in [
            [15, 'right'],
            [8, 'left'],
        ]:
            p = kinect_joints[hand_index]
            p_cube = get_cube(p)
            p_cube_proj = depth3d_to_color2d(p_cube, apud.cam_params[sn])
            crop_img = pil_crop(color_undistorted, get_bbox(p_cube_proj))
            # cv2.imwrite(osp.join(output_dirpath, f'{frame_index:06d}_{side}.png'), crop_img)
            theta_mano = processor.process(crop_img, side)
            result[frame_index][side] = deepcopy(theta_mano)
        done += 1
        if done % 100 == 0:
            print(f'done {done}, {time.time() - start}')
    with open(osp.join(output_dirpath, f'{pid}.pickle'), 'wb') as f:
        pickle.dump(result, f)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--pids', required=True)
    parser.add_argument('--sns', required=True)
    args = parser.parse_args()
    pids = args.pids.split(',')
    sns = args.sns.split(',')
    print('processing pids', pids)
    for pid in pids:
        for sn in sns:
            try:
                process_pid(pid, sn)
            except Exception as e:
                print(f'pid: {pid}, sn: {sn} exception:', e)
