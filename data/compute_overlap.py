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
            根据指定的filepath加载.ply文件，创建Cloud对象
        Args:
            filepath: .ply的路径

        Returns: Cloud对象

        '''
        # 加载.npz文件，（点，索引）
        arrays = np.load(filepath)
        # 返回Cloud对象
        return cls(arrays['points'], arrays['indices'])

    @classmethod
    def downsample_from(cls, pcd, max_points):
        '''
            对点云进行将降采样到指定最大的点数量
        Args:
            pcd: 要进行降采样的点云
            max_points: 如果len(points) > max_points 那么 max_points 否则 len(points)

        Returns:
            降采样之后的Cloud对象（降采样之后的点，在原点云中点的索引）
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
        in_root,                # fusion_dataset路径
        out_root,               # downsample路径
        max_points              # 保留点的最大数量（大于max_points，随机选取max_points个点，小于max_points，返回原来的点）
):
    import open3d as o3d
    o3d.utility.set_verbosity_level(o3d.utility.VerbosityLevel.Error)

    # 创建输出文件夹（log_correspond_points）
    uio.may_create_folder(out_root)
    # 获取所有.ply的名称
    pcd_names = uio.list_files(in_root, 'cloud_bin_*.ply', alphanum_sort=True)
    pcd_stems = list()
    for pname in pcd_names:
        # cloud_bin_*
        pstem = pname[:-4]
        # 当前.ply文件的名称
        pcd_path = osp.join(in_root, pname)
        # 当前.pose文件的名称
        # pose_path = osp.join(in_root, pstem + '.pose.npy')
        pose_path = osp.join(in_root, pstem + '.info.txt')
        pcd = o3d.io.read_point_cloud(pcd_path)
        # pose = np.load(pose_path)
        # pose = np.loadtxt(pose_path,skiprows=1)
        # 应用pose到点云上
        # pcd.transform(pose)
        # 对当前的点云执行降采样
        down_pcd = Cloud.downsample_from(pcd, max_points)
        # 保存降采样之后的点云（后续用于寻找匹配点）（非数组格式，使用.npz保存）（降采样之后的点，在原数组的点索引）
        down_pcd.save(osp.join(out_root, pstem + '.npz'))
        # 列表保存执行降采样的点云名称（cloud_bin_*）
        pcd_stems.append(pstem)

    return pcd_stems

# 计算重叠率
def compute_overlap(
        cfg,                    # config对象
        scene,                  # scene文件夹名称
        seq,                    # seq文件夹名称
        pcd_names,              # 所有经过降采样之后的cloud_bin_*名称
        p_index,                # 当前cloud_bin_*的id
        dist_thresh=0.075       # 对应点的距离阈值
):


    import pyflann

    # 执行降采样保存点云的文件夹路径
    downsample_folder = osp.join(cfg.temp_root, scene, seq)

    # 符合对应点条件保存的对应点索引的文件夹路径
    correspond_points_folder = osp.join(cfg.out_root, scene, seq)

    # 获取所有指定降采样的点云名称（cloud_bin_*）
    n_pcds = len(pcd_names)
    # 根据当前的获cloud_bin_*取Cloud对象 ( 降采样之后的点云，索引 )
    pcd_src = Cloud.load_from(osp.join(downsample_folder, pcd_names[p_index] + '.npz'))
    # 获取当前Cloud对象中的点数量
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
        '''
            knn_indices : 匹配点的索引,p_index
            knn_dists2  : 匹配点的距离
        '''
        knn_indices, knn_dists2 = kdtree_src.nn_index(pcd_dst.points,
                                                      num_neighbors=1,
                                                      checks=params_src['checks'])
        pair_indices = np.stack((knn_indices,range(len(pcd_dst.points))), axis=1)
        corr_indices = pair_indices[np.sqrt(knn_dists2) <= dist_thresh, :]
        # 获取重叠率
        overlap_ratio = float(len(corr_indices)) / max(n_points_src, n_points_dst)
        # 如果 重叠率 > 30% 那么 保存.ply_P 和 .ply_Q 的索引 else 跳过此次循环
        if overlap_ratio < 0.3:
            continue

        '''
            保存符合距离阈值的对应点索引数组
                cloud_bin_{p_index}_cloud_bin_{q_index}
                [[p_index,q_index],...]
        '''
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
# 执行seq文件夹
def run_seq(cfg, scene, seq):
    print("    Start {}".format(seq))


    # 获取降采样之后所有.ply文件的名称列表（该函数中，保存了.npz文件，（点，索引））
    pcd_names = downsample_pcds(
        osp.join(cfg.dataset_root, scene, seq),
        osp.join(cfg.temp_root, scene, seq),
        cfg.max_points
    )

    # 计算所有降采样点云的数量（其实和.ply文件的数量是一致的）
    n_pcds = len(pcd_names)

    # 创建保存对应点文件夹
    correspond_points_folder = osp.join(cfg.out_root, scene, seq)
    if osp.exists(correspond_points_folder):
        print('    Skip...')
        return
    uio.may_create_folder(correspond_points_folder)

    # 并行运行代码
    if cfg.threads > 1:
        from joblib import Parallel, delayed
        import multiprocessing
        # 并行运行compute_overlap() (3个任务同时进行)
        Parallel(n_jobs=cfg.threads)(
            delayed(compute_overlap)(cfg, scene, seq, pcd_names, i) for i in range(n_pcds))
    else:
        for i in range(n_pcds):
            compute_overlap(cfg, scene, seq, pcd_names, i)

    print("    Finished {}".format(seq))

# 执行scene文件夹
def run_scene(cfg, sid, scene):
    print("  Start {}th scene {} ".format(sid, scene))
    # 获取当前的scene文件夹路径
    scene_folder = osp.join(cfg.dataset_root, scene)
    # 获取在当前scene中的seq文件夹路径
    seqs = uio.list_folders(scene_folder, alphanum_sort=True)
    print("  {} sequences".format(len(seqs)))
    for seq in seqs:
        # 执行seq文件夹
        run_seq(cfg, scene, seq)

    print("  Finished {}th scene {} ".format(sid, scene))

def run(cfg):
    print("Start iterating dataset")
    # 创建对应点文件夹
    uio.may_create_folder(cfg.out_root)
    # 获取所有的scene文件夹名称
    scenes = uio.list_folders(cfg.dataset_root, alphanum_sort=False)
    print("{} scenes".format(len(scenes)))
    for sid, scene in enumerate(scenes):
        # 执行scene文件夹
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
'''
    该文件读取经过fuse_fragments_3DMatch.py融合过的.ply文件，
    保存降采样之后的{"points":[点],"indices":[索引]}，以及覆盖率超过30%的.ply文件的对应点索引
        目录:log_downsample_maxpoints
            cloud_bin_*.npz                 :   保存在原点云中随机选取的最多100000个点，以及索引
                文件是个字典，{"points":[点],"indices":[索引]}
        目录:log_correspond_points_index
            cloud_bin_*-cloud_bin_*.npy     :   保存.ply点云重叠率（Q中与P中对应点数量占P的总数的比率）超过30%的对应点P和Q的索引
                文件是个数组，[[P_index,Q_index],...]
'''
if __name__ == '__main__':
    cfg = parse_args()
    run(cfg)
