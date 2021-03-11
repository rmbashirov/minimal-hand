

import os
import os.path as osp
import numpy as np
import argparse
import pickle
import cv2
from collections import defaultdict
import time

from transforms3d.axangles import axangle2mat


from skimage import io
from azure_people_unpacked import AzurePeopleUnpackedDataset

from PIL import Image
import config
from capture import OpenCVCapture
from hand_mesh import HandMesh
from kinematics import mpii_to_mano
from utils import OneEuroFilter, imresize
from wrappers import ModelPipeline
from utils import *

import open3d as o3d
import pygame


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


def simple_crop(image, bbox):
    left, top, right, bottom = list(map(int, bbox))
    return image[top:bottom, left:right]


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
        return theta_mano, frame


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--pid', required=True)
    args = parser.parse_args()

    print(f'{"=" * 20} processing {args.pid} {"=" * 20}')

    kinect_unpacked_dp = "/home/renat/Desktop/offline_processor2/align"
    kinect_unpacked_dp_bt = "/home/renat/Desktop/offline_processor2/align"
    output_dirpath = "/home/renat/Desktop/minimal-hand"
    sn = '000589692912'

    apud = AzurePeopleUnpackedDataset(
        osp.join(kinect_unpacked_dp, args.pid),
        serial_numbers=[sn]
    )
    apud.dataset_dirpath = osp.join(kinect_unpacked_dp_bt, args.pid)
    apud.read_bt()
    apud.dataset_dirpath = osp.join(kinect_unpacked_dp, args.pid)
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

    view_mat = axangle2mat([1, 0, 0], np.pi)  # align different coordinate systems
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


    start = time.time()
    for frame_index in sorted(fns_dict.keys())[1000:]:
        img_path = osp.join(color_dirpath, fns_dict[frame_index])
        color_undistorted = io.imread(img_path)[:, :, :3]

        try:
            kinect_joints = apud.get_bt_joints(frame_index, sn)
        except:
            print(f'no joints for frame {frame_index}')
            continue
        kinect_joints = np.array(kinect_joints) / 1000
        for hand_index, side in [
            [15, 'right'],
            # [8, 'left'],
        ]:
            p = kinect_joints[hand_index]
            p_cube = get_cube(p)
            p_cube_proj = depth3d_to_color2d(p_cube, apud.cam_params[sn])
            bbox = get_bbox(p_cube_proj)
            print(bbox)

            # crop_img = simple_crop(color_undistorted, bbox)
            crop_img = pil_crop(color_undistorted, bbox)

            print(crop_img.shape)

            theta_mano, frame_large = processor.process(crop_img, side)
            print(frame_large.shape)

            v = hand_mesh.set_abs_quat(theta_mano)
            v *= 2  # for better visualization
            v = v * 1000 + np.array([0, 0, 400])
            v = mesh_smoother.process(v)
            mesh.triangles = o3d.utility.Vector3iVector(hand_mesh.faces)
            mesh.vertices = o3d.utility.Vector3dVector(np.matmul(view_mat, v.T).T)
            mesh.paint_uniform_color(config.HAND_COLOR)
            mesh.compute_triangle_normals()
            mesh.compute_vertex_normals()
            viewer.update_geometry()
            viewer.poll_events()
            # viewer.capture_screen_image(osp.join(output_dirpath, "%06d.jpg" % i))

            display.blit(
                pygame.surfarray.make_surface(
                    np.transpose(imresize(crop_img, (window_size, window_size)), (1, 0, 2))
                ),
                (0, 0)
            )
            pygame.display.update()

            # if keyboard.is_pressed("esc"):
            #   break

            clock.tick(30)

            result[frame_index][side] = theta_mano
        done += 1
        if done % 100 == 0:
            print(f'done {done}, {time.time() - start}')
    with open(osp.join(output_dirpath, f'{args.pid}.pickle'), 'wb') as f:
        pickle.dump(result, f)


if __name__ == "__main__":
    main()
