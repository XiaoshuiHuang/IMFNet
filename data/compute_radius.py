from __future__ import print_function
from __future__ import division

from collections import namedtuple
from pathlib import Path
import argparse
import math
import numpy as np
import os.path as osp
import sys

ROOT_DIR = osp.abspath('../../')
if ROOT_DIR not in sys.path:
    sys.path.append(ROOT_DIR)

from utils import uio


# 计算当前.ply中点的邻居
def compute_radius(
        cfg,                    # config对象
        scene,                  # 当前scene文件夹名称
        seq,                    # 当期seq文件夹名称
        pcd_name                # 当前.ply文件名称
):
    import open3d as o3d
    o3d.utility.set_verbosity_level(o3d.utility.VerbosityLevel.Error)

    # KNN中的计算邻居的半径
    nn_radius = cfg.radius

    print('    {}'.format(pcd_name))

    # 获取当前.ply的点云对象
    pcd = o3d.io.read_point_cloud(osp.join(cfg.dataset_root, scene, seq, pcd_name))
    # 获取当前.ply文件的点数量
    num_points = len(pcd.points)
    # 根据当前.ply的点云对象创建KDTree
    kdtree = o3d.geometry.KDTreeFlann(pcd)

    radii = list()
    for i in range(num_points):
        '''
            search_radius_vector_3d()，给出指定距离，搜索指定点到当前KDTree所有点的符合距离的点
                k : 符合距离nn_radius的点的数量
                nn_indices : 符合条件的点，在原点云对象中，点的索引
                nn_dists2 : 符合条件的点，与指定点的距离
        '''
        [k, nn_indices, nn_dists2] = kdtree.search_radius_vector_3d(pcd.points[i], nn_radius)
        if k < 2:
            radii.append(0)
        else:
            nn_indices = np.asarray(nn_indices)

            # dist = mean(sqrt(nn_dist))/2
            nn_dists2 = np.asarray(nn_dists2)
            nn_dists = np.sqrt(nn_dists2[nn_indices != i])
            radius = np.mean(nn_dists) * 0.5

            # 当前锚点在指定半径内的点的距离
            radii.append(radius)

    radii = np.asarray(radii, dtype=np.float32)
    # 保存每个.ply文件中每个点到其他点的距离
    np.save(osp.join(cfg.dataset_root, scene, seq, pcd_name[:-4] + '.radius.npy'), radii)


# 执行seq文件夹
def run_seq(cfg, scene, seq):
    print("    Start {}".format(seq))

    # 获取当前scene文件夹下，当前seq文件夹下所有的.ply文件名称
    pcd_names = uio.list_files(osp.join(cfg.dataset_root, scene, seq),
                               '*.ply',
                               alphanum_sort=True)
    # 是否并行运行以下代码（cfg.threads表示并行运行的数量）
    if cfg.threads > 1:
        from joblib import Parallel, delayed
        import multiprocessing
        # 并行运行compute_radius()(同时运行cfg.threads个任务)
        Parallel(n_jobs=cfg.threads)(
            delayed(compute_radius)(cfg, scene, seq, pcd_name) for pcd_name in pcd_names)
    else:
        for pcd_name in pcd_names:
            compute_radius(cfg, scene, seq, pcd_name)

    print("    Finished {}".format(seq))


# 执行scene文件夹
def run_scene(cfg, sid, scene):
    print("  Start {}th scene {} ".format(sid, scene))
    # 获取当前的scene文件夹路径
    scene_folder = osp.join(cfg.dataset_root, scene)
    # 获取在当前scene下所有的seq文件夹路径法
    seqs = uio.list_folders(scene_folder, alphanum_sort=True)
    print("  {} sequences".format(len(seqs)))
    for seq in seqs:
        # 执行seq文件夹
        run_seq(cfg, scene, seq)

    print("  Finished {}th scene {} ".format(sid, scene))


def run(cfg):
    print("Start iterating dataset")
    # 获取所有的scene文件夹路径
    scenes = uio.list_folders(cfg.dataset_root, alphanum_sort=False)
    print("{} scenes".format(len(scenes)))
    for sid, scene in enumerate(scenes):
        # 执行scene文件夹
        run_scene(cfg, sid, scene)

    print("Finished iterating dataset")


def parse_args():
    # dataset_path = "../../../3DMatch_RGB/train"
    # dataset_path = "../../../3DMatch_RGB/val"
    dataset_path = "/DISK/qwt/datasets/fusiondatsets/an"
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_root', default=dataset_path)
    parser.add_argument('--radius', type=float, default=0.075)
    parser.add_argument('--threads', type=int, default=8)

    return parser.parse_args()


'''
    该文件用于计算，经过fuse_fragments_3DMatch.py融合过的.ply文件中点与其他点的距离
    使用KNN计算（KDTree存取方式）
        目录：3DMatch_fusion_dataset
            cloud_bin_*.radius.npy     :  是一个列表，保存了相同名称的.ply文件中，每个点到符合半径内其他点的计算距离（索引完全对应）
'''
if __name__ == '__main__':
    cfg = parse_args()
    run(cfg)