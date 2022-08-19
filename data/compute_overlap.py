from __future__ import print_function
from __future__ import division

from collections import namedtuple
from pathlib import Path
import argparse
import math
import numpy as np
import os.path as osp
import sys

ROOT_DIR = osp.abspath('../')
if ROOT_DIR not in sys.path:
    sys.path.append(ROOT_DIR)

from util import uio

PCDMeta = namedtuple('PCDMeta', ['name', 'cloud'])


class Cloud(object):

    def __init__(self, points, indices ,normals=None):
        self.points = points
        self.indices = indices
        self.normals = normals

    def save(self, filepath):
        np.savez(filepath, points=self.points, indices=self.indices, normals=self.normals)

    @classmethod
    def load_from(cls, filepath):
        '''
             loading the .ply files according to filepath
        Args:
            filepath: .ply path

        Returns: Cloud Object

        '''
        arrays = np.load(filepath)
        return cls(arrays['points'], arrays['indices'])

    @classmethod
    def downsample_from(cls, pcd, max_points):
        '''
            downsample
        Args:
            pcd: point cloud
            max_points: if len(points) > max_points  max_points else len(points)

        Returns:

        '''
        points = np.asarray(pcd.points)
        n_points = len(points)
        if n_points <= max_points:
            return cls(points.astype(np.float32), np.arange(n_points))
        else:
            indices = np.random.choice(n_points, max_points, replace=False)
            downsampled = points[indices, :].astype(np.float32)
            return cls(downsampled, indices)

# 执行降采样
def downsample_pcds(
        in_root,                # fusion_dataset path
        out_root,               # downsample apth
        max_points              # max points
):
    import open3d as o3d
    o3d.utility.set_verbosity_level(o3d.utility.VerbosityLevel.Error)

    uio.may_create_folder(out_root)
    pcd_names = uio.list_files(in_root, 'cloud_bin_*.ply', alphanum_sort=True)
    pcd_stems = list()
    for pname in pcd_names:
        # cloud_bin_*
        pstem = pname[:-4]
        pcd_path = osp.join(in_root, pname)
        # pose_path = osp.join(in_root, pstem + '.pose.npy')
        pose_path = osp.join(in_root, pstem + '.info.txt')
        pcd = o3d.io.read_point_cloud(pcd_path)
        # pose = np.load(pose_path)
        # pose = np.loadtxt(pose_path,skiprows=1)
        # pcd.transform(pose)
        down_pcd = Cloud.downsample_from(pcd, max_points)
        down_pcd.save(osp.join(out_root, pstem + '.npz'))
        pcd_stems.append(pstem)

    return pcd_stems

# 计算重叠率
def compute_overlap(
        cfg,                    # config
        scene,                  # scene name
        seq,                    # seq name
        pcd_names,              # cloud_bin_* file names(downsample)
        p_index,                # the id of cloud_bin_*
        dist_thresh=0.075       #
):


    import pyflann

    downsample_folder = osp.join(cfg.temp_root, scene, seq)
    correspond_points_folder = osp.join(cfg.out_root, scene, seq)
    n_pcds = len(pcd_names)
    pcd_src = Cloud.load_from(osp.join(downsample_folder, pcd_names[p_index] + '.npz'))
    n_points_src = len(pcd_src.points)
    index_src = int(pcd_names[p_index][10:])
    kdtree_src = pyflann.FLANN()
    params_src = kdtree_src.build_index(pcd_src.points, algorithm='kdtree', trees=4)

    for q_index in range(p_index + 1, n_pcds):
        pcd_dst = Cloud.load_from(osp.join(downsample_folder, pcd_names[q_index] + '.npz'))
        n_points_dst = len(pcd_dst.points)
        index_dst = int(pcd_names[q_index][10:])
        assert index_src < index_dst
        if index_src + 1 == index_dst:
            continue
        knn_indices, knn_dists2 = kdtree_src.nn_index(pcd_dst.points,
                                                      num_neighbors=1,
                                                      checks=params_src['checks'])
        pair_indices = np.stack((knn_indices,range(len(pcd_dst.points))), axis=1)
        corr_indices = pair_indices[np.sqrt(knn_dists2) <= dist_thresh, :]
        overlap_ratio = float(len(corr_indices)) / max(n_points_src, n_points_dst)
        if overlap_ratio < 0.3:
            continue
        np.save(
            osp.join(
                correspond_points_folder,
                '{}-{}.npy'.format(
                    pcd_names[p_index],
                    pcd_names[q_index]
                )
            ),
            corr_indices
        )
        overlap_txt = osp.join(correspond_points_folder,f"{pcd_names[p_index]}-{pcd_names[q_index]}-overlap.txt")
        with open(file=overlap_txt,mode="w") as f:
            f.write(f"{overlap_ratio}")

def run_seq(cfg, scene, seq):
    print("    Start {}".format(seq))

    pcd_names = downsample_pcds(
        osp.join(cfg.dataset_root, scene, seq),
        osp.join(cfg.temp_root, scene, seq),
        cfg.max_points
    )

    n_pcds = len(pcd_names)

    correspond_points_folder = osp.join(cfg.out_root, scene, seq)
    if osp.exists(correspond_points_folder):
        print('    Skip...')
        return
    uio.may_create_folder(correspond_points_folder)

    if cfg.threads > 1:
        from joblib import Parallel, delayed
        import multiprocessing
        Parallel(n_jobs=cfg.threads)(
            delayed(compute_overlap)(cfg, scene, seq, pcd_names, i) for i in range(n_pcds))
    else:
        for i in range(n_pcds):
            compute_overlap(cfg, scene, seq, pcd_names, i)

    print("    Finished {}".format(seq))

def run_scene(cfg, sid, scene):
    print("  Start {}th scene {} ".format(sid, scene))
    scene_folder = osp.join(cfg.dataset_root, scene)
    seqs = uio.list_folders(scene_folder, alphanum_sort=True)
    print("  {} sequences".format(len(seqs)))
    for seq in seqs:
        run_seq(cfg, scene, seq)

    print("  Finished {}th scene {} ".format(sid, scene))

def run(cfg):
    print("Start iterating dataset")
    uio.may_create_folder(cfg.out_root)
    scenes = uio.list_folders(cfg.dataset_root, alphanum_sort=False)
    print("{} scenes".format(len(scenes)))
    for sid, scene in enumerate(scenes):
        run_scene(cfg, sid, scene)

    print("Finished iterating dataset")


def parse_args():
    # data_path = "/DISK/qwt/datasets/fusiondataset/an"
    # data_path = "/home/qwt/code/3DMatch_RGB/train"
    # data_path = "/DISK/qwt/datasets/fusiondataset/RGB3DMatch_train"
    # data_path = "/DISK/qwt/datasets/fusiondataset/downsample_0_025"
    data_path = "/DISK/qwt/datasets/Ours_train_0_01/train"
    downsample_path = "/DISK/qwt/datasets/Ours_train_0_01/downsample_maxpool"
    keypoints_path = "/DISK/qwt/datasets/Ours_train_0_01/keypoints"

    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_root', default=data_path)
    parser.add_argument('--temp_root', default=downsample_path)
    parser.add_argument('--out_root', default=keypoints_path)
    parser.add_argument('--max_points', type=int, default=300000)
    parser.add_argument('--threads', type=int, default=3)

    return parser.parse_args()

if __name__ == '__main__':
    cfg = parse_args()
    run(cfg)
