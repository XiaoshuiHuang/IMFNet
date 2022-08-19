from __future__ import division
from __future__ import print_function

from collections import defaultdict,namedtuple
from pathlib import Path
import cv2
import json
import numpy as np
import os
import os.path as osp
import re
import shutil
from nibabel import quaternions as nq
from typing import Tuple, List, Optional, Union, Any
import torch


def process_image(image, aim_H=480, aim_W=640, mode="resize", clip_mode="center"):

    H, W, C = np.array(image).shape

    '''
        H x W
        min:(1513,2141)
        max:(339,396)
    '''

    if (H == aim_H and W == aim_W):
        return np.array(image)

    if (mode == "resize"):
        # dsize = （W，H）
        image = np.asarray(
            cv2.resize(
                image,
                dsize=(aim_W, aim_H),
                interpolation=cv2.INTER_LINEAR
            ),
            dtype=np.float32
        )


    elif (mode == "clip"):

        while (H < aim_H or W < aim_W):
            image = cv2.pyrUp(src=image)
            H, W, C = np.array(image).shape

        if (H > aim_H * 2 and W > aim_W * 2):
            image = cv2.pyrDown(src=image)
            H, W, C = np.array(image).shape

        if (clip_mode == "center"):
            H_top = int((H - aim_H) / 2)
            W_left = int((W - aim_W) / 2)
            image = image[H_top:H_top + aim_H, W_left:W_left + aim_W]
        elif (clip_mode == "normal"):
            image = image[0:aim_H, 0:aim_W]
        elif (clip_mode == "random"):
            H_top = int(np.random.random() * (H - aim_H))
            W_left = int(np.random.random() * (W - aim_W))
            image = image[H_top:H_top + aim_H, W_left:W_left + aim_W]

    elif (mode == "padding"):
        # (C,H,W)
        image = np.transpose(image, (2, 0, 1))

        if(aim_H < H and aim_W < W):
            padding_H = aim_H - H
            padding_W = aim_W - W

            padding_H0 = np.zeros((C, padding_H, W))
            padding_W0 = np.zeros((C, aim_H, padding_W))

            image = np.concatenate([image, padding_H0], axis=1)
            image = np.concatenate([image, padding_W0], axis=2)
        elif(aim_H < H):
            image = image[:,0:aim_H,:]

            padding_W = aim_W - W
            padding_W0 = np.zeros((C, aim_H, padding_W))

            image = np.concatenate([image,padding_W0],axis=2)

        elif(aim_W < W):
            image = image[:,:,0:aim_W]

            padding_H = aim_H - H
            padding_H0 = np.zeros((C, padding_H, W))

            image = np.concatenate([image, padding_H0], axis=1)
        else:

            image = image[:,0:aim_H,0:aim_W]


        image = np.transpose(image,(1,2,0))

    return image


def apply_transform(points: np.ndarray, transform: np.ndarray, normals: Optional[np.ndarray] = None):
    rotation = transform[:3, :3]
    translation = transform[:3, 3]
    points = np.matmul(points, rotation.T) + translation
    if normals is not None:
        normals = np.matmul(normals, rotation.T)
        return points, normals
    else:
        return points

def compute_inlier_ratio(ref_corr_points, src_corr_points, transform, positive_radius=0.1):
    r"""Computing the inlier ratio between a set of correspondences."""
    '''
        ref_corr_points : es points
        src_corr_points : src points
        transform : GT
    '''
    src_corr_points = apply_transform(src_corr_points, transform)
    residuals = np.sqrt(((ref_corr_points - src_corr_points) ** 2).sum(1))
    inlier_ratio = np.mean(residuals < positive_radius)
    return inlier_ratio

def evaluate_correspondences(ref_points, src_points, transform, positive_radius=0.1):
    inlier_ratio = compute_inlier_ratio(ref_points, src_points, transform, positive_radius=positive_radius)

    return inlier_ratio

def compute_relative_rotation_error(gt_rotation: np.ndarray, est_rotation: np.ndarray):
    r"""Compute the isotropic Relative Rotation Error.

    RRE = acos((trace(R^T \cdot \bar{R}) - 1) / 2)

    Args:
        gt_rotation (array): ground truth rotation matrix (3, 3)
        est_rotation (array): estimated rotation matrix (3,     rre:2.5962121573077144,rte:0.08104451572937497,rr:0.8019986915430263)

    Returns:
        rre (float): relative rotation error.
    """
    x = 0.5 * (np.trace(np.matmul(est_rotation.T, gt_rotation)) - 1.0)
    x = np.clip(x, -1.0, 1.0)
    x = np.arccos(x)
    rre = 180.0 * x / np.pi
    return rre

def compute_relative_translation_error(gt_translation: np.ndarray, est_translation: np.ndarray):
    r"""Compute the isotropic Relative Translation Error.

    RTE = \lVert t - \bar{t} \rVert_2

    Args:
        gt_translation (array): ground truth translation vector (3,)
        est_translation (array): estimated translation vector (3,)

    Returns:
        rte (float): relative translation error.
    """
    return np.linalg.norm(gt_translation - est_translation)

def compute_registration_error(gt_transform: np.ndarray, est_transform: np.ndarray):
    r"""Compute the isotropic Relative Rotation Error and Relative Translation Error.

    Args:
        gt_transform (array): ground truth transformation matrix (4, 4)
        est_transform (array): estimated transformation matrix (4, 4)

    Returns:
        rre (float): relative rotation error.
        rte (float): relative translation error.
    """
    gt_rotation, gt_translation = get_rotation_translation_from_transform(gt_transform)
    est_rotation, est_translation = get_rotation_translation_from_transform(est_transform)
    rre = compute_relative_rotation_error(gt_rotation, est_rotation)
    rte = compute_relative_translation_error(gt_translation, est_translation)
    return rre, rte

def get_rotation_translation_from_transform(transform: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    r"""Get rotation matrix and translation vector from rigid transform matrix.

    Args:
        transform (array): (4, 4)

    Returns:
        rotation (array): (3, 3)
        translation (array): (3,)
    """
    rotation = transform[:3, :3]
    translation = transform[:3, 3]
    return rotation, translation

def compute_transform_error(transform, covariance, estimated_transform):
    relative_transform = np.matmul(np.linalg.inv(transform), estimated_transform)
    R, t = get_rotation_translation_from_transform(relative_transform) # tor trans
    q = nq.mat2quat(R)
    er = np.concatenate([t, q[1:]], axis=0)
    p = er.reshape(1, 6) @ covariance @ er.reshape(6, 1) / covariance[0, 0]
    return p.item()

Pose = namedtuple('Pose', ['indices', 'transformation'])

def read_log(filepath):
    lines = read_lines(filepath)
    n_poses = len(lines) // 5
    poses = list()
    for i in range(n_poses):
        items = lines[i * 5].split()  # Meta line
        id0, id1, id2 = int(items[0]), int(items[1]), int(items[2])
        mat = np.zeros((4, 4), dtype=np.float64)
        for j in range(4):
            items = lines[i * 5 + j + 1].split()
            for k in range(4):
                mat[j, k] = float(items[k])
        poses.append(Pose(indices=[id0, id1, id2], transformation=mat))
    return poses

def read_info_file(file_name):
    with open(file_name) as f:
        lines = f.readlines()
        lines = [line.strip() for line in lines]
    test_pairs = []
    num_pairs = len(lines) // 7
    for i in range(num_pairs):
        line_id = i * 7
        split_line = lines[line_id].split()
        test_pair = [int(split_line[0]), int(split_line[1])]
        num_fragments = int(split_line[2])
        info = []
        for j in range(1, 7):
            info.append(lines[line_id + j].split())
        info = np.array(info, dtype=np.float32)
        test_pairs.append(dict(test_pair=test_pair, num_fragments=num_fragments, covariance=info))
    return test_pairs

def read_keypoints(filepath):
    return np.load(filepath)


def read_descriptors(desc_type, root_dir, scene_name, seq_name, pcd_name):
    filepath = osp.join(root_dir, scene_name, seq_name, pcd_name + '.desc.npy')
    descs = np.load(filepath)
    return descs


def knn_search(points_src, points_dst, k=1):

    import open3d as o3d
    o3d.utility.set_verbosity_level(o3d.utility.VerbosityLevel.Error)

    kdtree = o3d.geometry.KDTreeFlann(np.asarray(points_dst.T, dtype=np.float64))
    points_src = np.asarray(points_src, dtype=np.float64)
    nnindices = [
        kdtree.search_knn_vector_xd(points_src[i, :], k)[1] for i in range(len(points_src))
    ]
    if k == 1:
        return np.asarray(nnindices, dtype=np.int32)[:, 0]
    else:
        return np.asarray(nnindices, dtype=np.int32)

def imageOfPoint(
        points,
        points_all,
        intrinsic,
        images,
        image_size,
        image_same=False
):
    image_list = []

    if(image_same):
        for point in points:
            image = p2i(
                point=point,
                points_all=points_all,
                intrinsic=intrinsic,
                image=images,
                image_size=image_size
            )
            image_list.append(image)
    else:
        assert len(points) == len(images)

        for point,image in zip(points,images):
            image = p2i(
                point=point,
                points_all=points_all,
                intrinsic=intrinsic,
                image=image,
                image_size=image_size
            )
            image_list.append(image)

    image_list = np.array(image_list)
    #print(f"shape:{image_list.shape}")
    image_list = np.concatenate(image_list,axis=0)

    # 返回切分图片
    return image_list

def p2i(point,points_all,intrinsic,image,image_size):

    height, width, _ = image.shape
    x, y = carema2pixe(
        point=point,
        points_all=points_all,
        intrinsic=intrinsic,
        W=width,
        H=height
    )

    # composate
    if(height < image_size or width < image_size):
        image_temp = image.copy()
        while (height < image_size):
            image = np.concatenate([image, image_temp], axis=0)
            height, _, _ = image.shape
        image_temp = image.copy()
        while (width < image_size):
            image = np.concatenate([image, image_temp], axis=1)
            _, width, _ = image.shape
        center_h = (height - image_size) / 2
        center_w = (width - image_size) / 2
        image = image[
                int(center_h):int(image_size+center_h),
                int(center_w):int(image_size+center_w),
                :
        ]
        image = np.expand_dims(np.transpose(image,axes=(2,0,1)),axis=0)
        print("full up image!")
        return image

    image_size = image_size // 2


    if (x - image_size < 0 or x + image_size > width or y - image_size < 0 or y + image_size > height):
        if (x - image_size < 0 and y - image_size < 0):
            image_middle = image[
                0:np.round(y + image_size).astype(int),
                0:np.round(x + image_size).astype(int),
                :
            ]
            image_x = image[
                0:np.round(y + image_size).astype(int),
                np.array(width - (image_size - x)).astype(int):width,
                :
            ]
            image_x = np.concatenate([image_x, image_middle], axis=1)
            image_y = image[
                np.array(height - (image_size - y)).astype(int):height,
                0:2 * image_size,
                :
            ]
            image = np.concatenate([image_y, image_x], axis=0)
        elif (x - image_size < 0 and y + image_size > height):
            image_middle = image[
                np.round(y - image_size).astype(int):height,
                0:np.round(x + image_size).astype(int),
                :
            ]
            image_x = image[
                np.round(y - image_size).astype(int):height,
                np.array(width - (image_size - x)).astype(int):width,
                :
            ]
            image_x = np.concatenate([image_x, image_middle], axis=1)
            image_y = image[
                0:np.round((y + image_size) - height).astype(int),
                0:2 * image_size,
                :
            ]
            image = np.concatenate([image_x, image_y], axis=0)
        elif (x + image_size > width and y + image_size > height):
            image_middle = image[
                np.round(y - image_size).astype(int):height,
                np.round(x - image_size).astype(int):width,
                :
            ]
            image_x = image[
                np.round(y - image_size).astype(int):height,
                0:np.array((image_size + x) - width).astype(int),
                :
            ]
            image_x = np.concatenate([image_middle, image_x], axis=1)
            image_y = image[
                0:np.array((y + image_size) - height).astype(int),
                width - 2 * image_size:width,
                :
            ]
            image = np.concatenate([image_x, image_y], axis=0)
        elif (x + image_size > width and y - image_size < 0):
            image_middle = image[
                0:np.round(y + image_size).astype(int),
                np.round(x - image_size).astype(int):width,
                :
            ]
            image_x = image[
                0:np.round(y + image_size).astype(int),
                0:np.array((image_size + x) - width).astype(int),
                :
            ]
            image_x = np.concatenate([image_middle, image_x], axis=1)
            image_y = image[
                np.array(height - (image_size - y)).astype(int):height,
                width - 2 * image_size:width,
                :
            ]
            image = np.concatenate([image_y, image_x], axis=0)
        elif (x - image_size < 0 and y - image_size >= 0 and y + image_size <= height):
            image_middle = image[
                np.round(y - image_size).astype(int):np.round(y + image_size).astype(int),
                0:np.round(x + image_size).astype(int),
                :
            ]
            image_x = image[
                np.round(y - image_size).astype(int):np.round(y + image_size).astype(int),
                np.array(width - (image_size - x)).astype(int):width,
                :
            ]
            image = np.concatenate([image_x, image_middle], axis=1)
        elif (x + image_size > width and y - image_size >= 0 and y + image_size <= height):
            image_middle = image[
               np.round(y - image_size).astype(int):np.round(y + image_size).astype(int),
               np.round(x - image_size).astype(int):width,
               :
            ]
            image_x = image[
                np.round(y - image_size).astype(int):np.round(y + image_size).astype(int),
                0:np.array((x + image_size) - width).astype(int),
                :
            ]
            image = np.concatenate([image_middle, image_x], axis=1)
        elif (y - image_size < 0 and x - image_size >= 0 and x - image_size <= width):
            image_middle = image[
                0:np.round(y + image_size).astype(int),
                np.round(x - image_size).astype(int):np.round(x + image_size).astype(int),
                :
            ]
            image_y = image[
                np.round(height - (image_size - y)).astype(int):height,
                np.round(x - image_size).astype(int):np.round(x + image_size).astype(int),
                :
            ]
            image = np.concatenate([image_y, image_middle], axis=0)
        elif (y + image_size > height and x - image_size >= 0 and x + image_size <= width):
            image_middle = image[
                np.round(y - image_size).astype(int):height,
                np.round(x - image_size).astype(int):np.round(x + image_size).astype(int),
                :
            ]
            image_y = image[
                0:np.round((y + image_size) - height).astype(int),
                np.round(x - image_size).astype(int):np.round(x + image_size).astype(int),
                :
            ]
            image = np.concatenate([image_middle, image_y], axis=0)
    else:
        image = image[
            np.round(y - image_size).astype(int):np.round(y + image_size).astype(int),
            np.round(x - image_size).astype(int):np.array(x + image_size).astype(int),
            :
        ]

    image = np.expand_dims(np.transpose(image,axes=(2,0,1)),axis=0)
    # print(image.shape)

    return image

def max_pixel(points,intrinsic):

    CAM_FX, CAM_FY = intrinsic[0,0], intrinsic[1,1]  # fx/fy
    CAM_CX, CAM_CY = intrinsic[0,2], intrinsic[1,2]  # cx/cy

    EPS = 1.0e-16

    valid = points[:, 2] > EPS
    z = points[valid, 2]

    u = points[valid, 0] * CAM_FX / z + CAM_CX  # x
    v = points[valid, 1] * CAM_FY / z + CAM_CY  # y

    return np.max(abs(u)), np.max(abs(v))

def carema2pixe(point,points_all,intrinsic,W=480,H=640):

    CAM_FX, CAM_FY = intrinsic[0, 0], intrinsic[1, 1]  # fx/fy
    CAM_CX, CAM_CY = intrinsic[0, 2], intrinsic[1, 2]  # cx/cy

    x, y, z = point

    u = abs(x * CAM_FX / z + CAM_CX ) # x
    v = abs(y * CAM_FY / z + CAM_CY ) # y

    U_MAX, V_MAX = max_pixel(points=points_all, intrinsic=intrinsic)

    U_scale = W / U_MAX
    V_scale = H / V_MAX

    u = np.floor(u * U_scale).astype(int)
    v = np.floor(v * V_scale).astype(int)

    u = np.round(u).astype(int)
    v = np.round(v).astype(int)
    # print(f"u:{u},v:{v}")

    # (W,H)
    return (u, v)

def check_carema2pixes(points,intrinsic):

    for point in points:
        u,v = check_carema2pixe(point,intrinsic)
        # print(f"u:{u},v:{v}")
        if(u < 0 or v < 0):
            return False
    return True

def check_carema2pixe(point,intrinsic):

    CAM_FX, CAM_FY = intrinsic[0, 0], intrinsic[1, 1]  # fx/fy
    CAM_CX, CAM_CY = intrinsic[0, 2], intrinsic[1, 2]  # cx/cy

    x, y, z = point

    u = x * CAM_FX / z + CAM_CX  # x
    v = y * CAM_FY / z + CAM_CY  # y

    u = np.round(u).astype(int)
    v = np.round(v).astype(int)

    # (W,H)
    return (u, v)


def is_number(s):
    try:
        float(s)
        return True
    except ValueError:
        return False

# create the specified folder
def may_create_folder(folder_path):
    if not osp.exists(folder_path):
        oldmask = os.umask(000)
        os.makedirs(folder_path, mode=0o777)
        os.umask(oldmask)
        return True
    return False


def make_clean_folder(folder_path):
    success = may_create_folder(folder_path)
    if not success:
        shutil.rmtree(folder_path)
        may_create_folder(folder_path)


def sorted_alphanum(file_list_ordered):
    convert = lambda text: int(text) if text.isdigit() else text
    alphanum_key = lambda key: [convert(c) for c in re.split('([0-9]+)', key) if len(c) > 0]
    return sorted(file_list_ordered, key=alphanum_key)


def list_files(folder_path, name_filter, alphanum_sort=False):
    file_list = [p.name for p in list(Path(folder_path).glob(name_filter))]
    if alphanum_sort:
        return sorted_alphanum(file_list)
    else:
        return sorted(file_list)


def list_folders(folder_path, name_filter=None, alphanum_sort=False):
    folders = list()
    for subfolder in Path(folder_path).iterdir():
        if subfolder.is_dir() and not subfolder.name.startswith('.'):
            folder_name = subfolder.name
            if name_filter is not None:
                if name_filter in folder_name:
                    folders.append(folder_name)
            else:
                folders.append(folder_name)
    if alphanum_sort:
        return sorted_alphanum(folders)
    else:
        return sorted(folders)


def read_lines(file_path):
    """
    :param file_path:
    :return:
    """
    with open(file_path, 'r') as fin:
        lines = [line.strip() for line in fin.readlines() if len(line.strip()) > 0]
    return lines


def read_json(filepath):
    with open(filepath, 'r') as fh:
        ret = json.load(fh)
    return ret


def last_log_folder(root_folder, prefix, digits=3):
    prefix_len = len(prefix)
    tmp = list()
    for folder in list_folders(root_folder, alphanum_sort=True):
        if not folder.startswith(prefix):
            continue
        assert not is_number(folder[prefix_len + digits])
        tmp.append((int(folder[prefix_len:prefix_len + digits]), folder))
    if len(tmp) == 0:
        return 0, None
    else:
        tmp = sorted(tmp, key=lambda tup: tup[0])
        return tmp[-1][0], tmp[-1][1]


def new_log_folder(root_folder, prefix, digits=3):
    idx, _ = last_log_folder(root_folder, prefix, digits)
    tmp = prefix + '{:0' + str(digits) + 'd}'
    assert idx + 1 < 10**digits
    return tmp.format(idx + 1)


def last_checkpoint(root_folder, prefix):
    tmp = defaultdict(list)
    for file in list_files(root_folder, '{}*.pth'.format(prefix), alphanum_sort=True):
        stem = file[:-4]
        values = stem.split('_')
        tmp[values[1]].append(int(values[-1]))
    for k, v in tmp.items():
        return prefix + '_{}_' + str(sorted(v)[-1]) + '.pth'


def read_color_image(file_path):
    img = cv2.imread(file_path)
    return img[..., ::-1]


def read_gray_image(file_path):
    img = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)
    return img


def read_16bit_image(file_path):
    img = cv2.imread(file_path, cv2.IMREAD_UNCHANGED)
    return img


def write_color_image(file_path, image):
    cv2.imwrite(file_path, image[..., ::-1])
    return file_path


def write_gray_image(file_path, image):
    cv2.imwrite(file_path, image)
    return file_path


def write_image(file_path, image):
    if image.ndim == 2:
        return write_gray_image(file_path, image)
    elif image.ndim == 3:
        return write_color_image(file_path, image)
    else:
        raise RuntimeError('Image dimensions are not correct!')

def read_pcds(root_folder, transform):
    import open3d as o3d

    ret = dict()
    for pcd_name in list_files(root_folder, '*.pcd', alphanum_sort=True):
        pcd_path = osp.join(root_folder, pcd_name)
        pcd_stem = pcd_name[:-4]
        pcloud = o3d.io.read_point_cloud(pcd_path) 
        if transform: 
            vp_path = osp.join(root_folder, pcd_stem + '.vp.json')
            vparams = read_json(vp_path)
            modelview = np.asarray(vparams['modelview_matrix'], np.float32)
            modelview = np.reshape(modelview, (4, 4)).T
            modelview_inv = np.linalg.inv(modelview)
            pcloud.transform(modelview_inv)
        ret[pcd_stem] = np.asarray(pcloud.points)
    return ret


def write_correspondence_ply(file_path,
                             pcloudi,
                             pcloudj,
                             edges,
                             colori=(255, 255, 0),
                             colorj=(255, 0, 0),
                             edge_color=(255, 255, 255)):
    num_pointsi = len(pcloudi)
    num_pointsj = len(pcloudj)
    num_points = num_pointsi + num_pointsj
    with open(file_path, 'w') as fh:
        fh.write('ply\n')
        fh.write('format ascii 1.0\n')
        fh.write('element vertex {}\n'.format(num_points))
        fh.write('property float x\n')
        fh.write('property float y\n')
        fh.write('property float z\n')
        fh.write('property uchar red\n')
        fh.write('property uchar green\n')
        fh.write('property uchar blue\n')
        fh.write('element edge {}\n'.format(len(edges)))
        fh.write('property int vertex1\n')
        fh.write('property int vertex2\n')
        fh.write('property uchar red\n')
        fh.write('property uchar green\n')
        fh.write('property uchar blue\n')
        fh.write('end_header\n')

        for k in range(num_pointsi):
            fh.write('{} {} {} {} {} {}\n'.format(pcloudi[k, 0], pcloudi[k, 1], pcloudi[k, 2],
                                                  colori[0], colori[1], colori[2]))
        for k in range(num_pointsj):
            fh.write('{} {} {} {} {} {}\n'.format(pcloudj[k, 0], pcloudj[k, 1], pcloudj[k, 2],
                                                  colorj[0], colorj[1], colorj[2]))
        for k in range(len(edges)):
            fh.write('{} {} {} {} {}\n'.format(edges[k][0], edges[k][1] + num_pointsi,
                                               edge_color[0], edge_color[1], edge_color[2]))
