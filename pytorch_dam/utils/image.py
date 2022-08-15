import cv2
import numpy as np
import torch
from torchvision.transforms import Compose, Normalize, ToTensor
import open3d as o3d
import os
import MinkowskiEngine as ME
from util.uio import process_image
from util.pointcloud import make_open3d_point_cloud

import matplotlib.image as img

# generate the PC tensor
def get_METensor(
        pc_path,
        image_path,
        image_H,
        image_W,
        voxel_size,
        device
):

    # 读取点云文件
    pcd = o3d.io.read_point_cloud(pc_path)
    xyz = np.asarray(pcd.points)

    # read image
    image = img.imread(image_path)
    if (image.shape[0] != image_H or image.shape[1] != image_W):
        image = process_image(image=image, aim_H=image_H, aim_W=image_W)
    image = np.transpose(image, axes=(2, 0, 1))
    image = np.expand_dims(image, axis=0)

    feats = []
    feats.append(np.ones((len(xyz), 1)))
    feats = np.hstack(feats)

    # Voxelize xyz and feats
    coords = np.floor(xyz /voxel_size)
    coords, inds = ME.utils.sparse_quantize(coords, return_index=True)
    # Convert to batched coords compatible with ME
    coords = ME.utils.batched_coordinates([coords])
    return_coords = xyz[inds]

    feats = feats[inds]

    feats = torch.tensor(feats, dtype=torch.float32)
    coords = torch.tensor(coords, dtype=torch.int32)

    device = torch.device('cuda' if device else 'cpu')

    stensor = ME.SparseTensor(feats, coordinates=coords, device=device)
    image = torch.as_tensor(image, dtype=torch.float32, device=device)

    return return_coords,stensor,image,xyz,inds


# 处理图片函数
def preprocess_image(img: np.ndarray, mean=None, std=None) -> torch.Tensor:
    if std is None:
        std = [0.5, 0.5, 0.5]
    if mean is None:
        mean = [0.5, 0.5, 0.5]

    preprocessing = Compose([
        ToTensor(),
        Normalize(mean=mean, std=std)
    ])

    return preprocessing(img.copy()).unsqueeze(0)

def deprocess_image(img):
    """ see https://github.com/jacobgil/keras-grad-cam/blob/master/grad-cam.py#L65 """
    img = img - np.mean(img)
    img = img / (np.std(img) + 1e-5)
    img = img * 0.1
    img = img + 0.5
    img = np.clip(img, 0, 1)
    return np.uint8(img * 255)

def show_cam_on_image(img: np.ndarray,
                      mask: np.ndarray,
                      use_rgb: bool = False,
                      colormap: int = cv2.COLORMAP_JET) -> np.ndarray:
    """ This function overlays the cam mask on the image as an heatmap.
    By default the heatmap is in BGR format.

    :param img: The base image in RGB or BGR format.
    :param mask: The cam mask.
    :param use_rgb: Whether to use an RGB or BGR heatmap, this should be set to True if 'img' is in RGB format.
    :param colormap: The OpenCV colormap to be used.
    :returns: The default image with the cam overlay.
    """
    heatmap = cv2.applyColorMap(np.uint8(255 * mask), colormap)
    if use_rgb:
        heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
    heatmap = np.float32(heatmap) / 255

    if np.max(img) > 1:
        raise Exception("The input image should np.float32 in the range [0, 1]")

    cam = heatmap + img
    cam = cam / np.max(cam)
    return np.uint8(255 * cam)

def get_Point_Cloud(points):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    return pcd

def normalization(data,a,b):
    data_min = np.min(data)
    data_max = np.max(data)

    k = (b-a)/(data_max - data_min)
    data = a + k * (data - data_min)

    return data

from pylab import cm
import matplotlib.pyplot as plt

def show_dam_on_pc_voxel_size(
        dam,
        points,
        target_point,
        knn=None
):
    if(not isinstance(points,o3d.geometry.PointCloud)):
        points = get_Point_Cloud(points)
    points.colors = o3d.utility.Vector3dVector(np.ones(shape=(len(points.points),3)))

    dam = normalization(dam, a=0.1, b=1)
    min_weight = np.min(dam)

    colors = np.asarray(cm.hsv(dam))[:, :3]
    for index,weight in enumerate(dam):
        if(weight == min_weight):
            np.asarray(points.colors)[index, :]= [0.5627450980392157,0.5627450980392157,0.5627450980392157]
            # np.asarray(points.colors)[index, :]= [0.7,0.7,0.7]
        else:
            np.asarray(points.colors)[index, :] = colors[index]

    np.asarray(points.colors)[target_point, :] = [0, 0, 0]
    if(knn):
        kdtree = o3d.geometry.KDTreeFlann(points)
        k,index,distance = kdtree.search_knn_vector_3d(
            query=points.points[target_point],
            knn=2
        )
        np.asarray(points.colors)[index, :] = [0, 0, 0]

        T=[
            [ 9.96926560e-01	 , 6.68735757e-02	 ,-4.06664421e-02	, -1.15576939e-01],
            [-6.61289946e-02	,  9.97617877e-01	,  1.94008687e-02,	 -3.87705398e-02],
            [ 4.18675510e-02,	 -1.66517807e-02	,  9.98977765e-01,	  1.14874890e-01],
            [ 0.00000000e+00	,  0.00000000e+00	,  0.00000000e+00	,  1.00000000e+00],
        ]
        points.transform(T)

    ply_path = "files/3D_head_map.ply"
    o3d.io.write_point_cloud(filename=ply_path,pointcloud=points)
    print(f"Saving : {ply_path}")

    o3d.visualization.draw_geometries(
        [points],
        window_name="dam visualization"
    )


