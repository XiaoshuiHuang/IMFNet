from __future__ import division
from __future__ import print_function

import warnings
warnings.filterwarnings("ignore")

import os
os.environ["CUDA_VISIBLE_DEVICES"]="0"
import argparse
import numpy as np
import os.path as osp
import sys
import pickle
import MinkowskiEngine as ME
import copy
import tqdm

from pathlib import Path

ROOT_DIR = osp.abspath('../')
if ROOT_DIR not in sys.path:
    sys.path.append(ROOT_DIR)

from util import uio
from util.pointcloud import make_open3d_point_cloud,make_open3d_feature_from_numpy
from scripts.benchmark_util import run_ransac

INLIER_THRESHES = [
    0.1,
]
# INLIER_RATIO_THRESHES = (np.arange(0, 21, dtype=np.float32) * 0.2 / 20).tolist()
INLIER_RATIO_THRESHES = [0.05,0.20]

VALID_SCENE_NAMES = []

TEST_SCENE_NAMES = [
    '7-scenes-redkitchen',
    'sun3d-home_at-home_at_scan1_2013_jan_1',
    'sun3d-home_md-home_md_scan9_2012_sep_30',
    'sun3d-hotel_uc-scan3',
    'sun3d-hotel_umd-maryland_hotel1',
    'sun3d-hotel_umd-maryland_hotel3',
    'sun3d-mit_76_studyroom-76-1studyroom2',
    'sun3d-mit_lab_hj-lab_hj_tea_nov_2_2012_scan1_erika',
]

TEST_SCENE_ABBR_NAMES = [
    'Kitchen',
    'Home_1',
    'Home_2',
    'Hotel_1',
    'Hotel_2',
    'Hotel_3',
    'Study',
    'MIT_Lab',
]

class RegisterResult(object):

    def __init__(
            self,
            frag1_name,
            frag2_name,
            num_inliers,
            inlier_ratio,
            gt_flag,
            rr=None,
            rre=None,
            rte=None,
            ir=None
    ):
        '''
        :param frag1_name: p
        :param frag2_name: q
        :param num_inliers:
        :param inlier_ratio:
        :param gt_flag:
        '''
        self.frag1_name = frag1_name
        self.frag2_name = frag2_name
        self.num_inliers = num_inliers
        self.inlier_ratio = inlier_ratio
        self.gt_flag = gt_flag
        self.rr = rr,
        self.rre = rre,
        self.rte = rte,
        self.ir = ir

def register_fragment_pair(
        scene_name,                     # current scene
        seq_name,                       # seq name (seq-01)
        frag1_name,                     # p -- id
        frag2_name,                     # q -- id
        desc_type,                      # IMFNet
        poses,                          # GT pose
        infos,                          # RMSE
        pcloud_root,                    # Testing Set path
        desc_root,                      # desc path
        inlier_thresh,                  # 0.1
        overlap_pid,                    # the id of pose and info
        cfg,
):
    '''
    :param scene_name:
    :param seq_name:
    :param frag1_name:
    :param frag2_name:
    :param desc_type:
    :param poses:
    :param pcloud_root:
    :param desc_root:
    :param inlier_thresh:
    :return:
        num_inliers         :              [0.1]
        inlier_ratio        :              [0.05,0.2]
        gt_flag             :              1: True，0: False
    '''
    import open3d as o3d
    o3d.utility.set_verbosity_level(o3d.utility.VerbosityLevel.Error)

    # point cloud -- id
    frag1_id = int(frag1_name.split('_')[-1])
    frag2_id = int(frag2_name.split('_')[-1])
    assert frag1_id < frag2_id

    num_rand_keypoints = cfg.num_rand_keypoints
    voxel_size = cfg.voxel_size

    data_i = np.load(osp.join(desc_root,scene_name,seq_name, frag1_name + ".npz"))
    coord_i, points_i, feat_i = data_i['xyz'], data_i['points'], data_i['feature']
    data_j = np.load(osp.join(desc_root,scene_name,seq_name, frag2_name+ ".npz"))
    coord_j, points_j, feat_j = data_j['xyz'], data_j['points'], data_j['feature']

    frag1_kpts = None
    frag2_kpts = None
    frag1_descs = None
    frag2_descs = None

    # random points
    if num_rand_keypoints > 0:
        # keypoints path
        keypoints_name = f"{scene_name}_{seq_name}_{frag1_id}_{frag2_id}_keypoints.npz"
        keypoints_folder = osp.join(cfg.out_root,desc_type+"_keypoints")
        uio.may_create_folder(keypoints_folder)
        keypoints_path = osp.join(keypoints_folder, keypoints_name)
        # use the keypoints in 3DMatch
        if(cfg.keypoints):
            # loding keypoints
            keypoints = np.load(keypoints_path)
            inds_i,inds_j = keypoints["inds_i"],keypoints["inds_j"]
            # print(f"Loding: {keypoints_path}")
        else:
            # sample points（5000）
            Ni, Nj = len(points_i), len(points_j)
            inds_i = np.random.choice(Ni, min(Ni, num_rand_keypoints), replace=False)
            inds_j = np.random.choice(Nj, min(Nj, num_rand_keypoints), replace=False)

            # save keypoints
            np.savez(inds_i=inds_i, inds_j=inds_j, file=keypoints_path)
            # print(f"Saving: {keypoints_path}")

        sample_i, sample_j = points_i[inds_i], points_j[inds_j]

        key_points_i = ME.utils.fnv_hash_vec(np.floor(sample_i / voxel_size))
        key_points_j = ME.utils.fnv_hash_vec(np.floor(sample_j / voxel_size))

        key_coords_i = ME.utils.fnv_hash_vec(np.floor(coord_i / voxel_size))
        key_coords_j = ME.utils.fnv_hash_vec(np.floor(coord_j / voxel_size))

        inds_i = np.where(np.isin(key_coords_i, key_points_i))[0]
        inds_j = np.where(np.isin(key_coords_j, key_points_j))[0]

        frag1_kpts, frag1_descs = coord_i[inds_i], feat_i[inds_i]
        frag2_kpts, frag2_descs = coord_j[inds_j], feat_j[inds_j]

    # ---------  RR  ---------
    covariance = infos[overlap_pid]['covariance']
    frag1_kpts_3d = make_open3d_point_cloud(frag1_kpts)
    feat_i_3d = make_open3d_feature_from_numpy(frag1_descs)
    frag2_kpts_3d = make_open3d_point_cloud(frag2_kpts)
    feat_j_3d = make_open3d_feature_from_numpy(frag2_descs)
    if len(frag1_kpts_3d.points) < len(frag2_kpts_3d.points):
        trans = run_ransac(frag1_kpts_3d, frag2_kpts_3d, feat_i_3d, feat_j_3d, cfg.voxel_size,ransac_n=3)
    else:
        trans = run_ransac(frag2_kpts_3d, frag1_kpts_3d, feat_j_3d, feat_i_3d, cfg.voxel_size,ransac_n=3)
        trans = np.linalg.inv(trans)
    es_T = np.linalg.inv(trans)
    gt_T = poses[overlap_pid].transformation
    error = uio.compute_transform_error(gt_T, covariance, es_T)
    accepted = error < 0.2 ** 2
    rr = 0
    rre = 0
    rte = 0
    if (accepted):
        rre, rte = uio.compute_registration_error(gt_T, es_T)
        rr = 1

    frag2_kpts_3d_es = copy.deepcopy(frag2_kpts_3d)
    frag2_kpts_3d_es.transform(es_T)
    ir = uio.evaluate_correspondences(
        np.asarray(frag2_kpts_3d_es.points), np.asarray(frag2_kpts_3d.points), gt_T, positive_radius=0.1
    )

    rs = [rr,rre,rte,ir]
    # ---------  RR  ---------

    frag21_nnindices = uio.knn_search(frag2_descs, frag1_descs)
    assert frag21_nnindices.ndim == 1
    frag12_nnindices = uio.knn_search(frag1_descs, frag2_descs)
    assert frag12_nnindices.ndim == 1

    frag2_match_indices = np.flatnonzero(
        np.equal(
            np.arange(len(frag21_nnindices)),
            frag12_nnindices[frag21_nnindices]
        )
    )


    frag2_match_kpts = frag2_kpts[frag2_match_indices, :]
    frag1_match_kpts = frag1_kpts[frag21_nnindices[frag2_match_indices], :]

    frag2_pcd_tmp = o3d.geometry.PointCloud()
    frag2_pcd_tmp.points = o3d.utility.Vector3dVector(frag2_match_kpts)
    frag2_pcd_tmp.transform(poses[overlap_pid].transformation)

    # metric
    distances = np.sqrt(
        np.sum(np.square(
            frag1_match_kpts - np.asarray(frag2_pcd_tmp.points)
        ), axis=1)
    )
    num_inliers = np.sum(distances < inlier_thresh)
    inlier_ratio = num_inliers / len(distances)
    gt_flag = 1
    return num_inliers, inlier_ratio, gt_flag, rs


def run_scene_matching(scene_name,                  # current scene
                       seq_name,                    # seq name（seq-01 in testing）
                       desc_type,                   # IMFNet
                       pcloud_root,                 # Testing Set path
                       desc_root,                   # desc path
                       out_root,                    # result path
                       inlier_thresh=0.1,           # 0.1
                       cfg=None,
                       ):

    # create the output files
    out_folder = osp.join(out_root, desc_type)
    uio.may_create_folder(out_folder)
    out_filename = '{}-{}-{:.2f}'.format(scene_name, seq_name, inlier_thresh)
    # skipping exist files
    if Path(osp.join(out_folder, out_filename + '.pkl')).is_file():
        print('[*] {} already exists. Skip computation.'.format(out_filename))
        return osp.join(out_folder, out_filename)

    # all cloud_bin_* files
    fragment_names = uio.list_files(
        osp.join(
            pcloud_root,
            scene_name,
            seq_name
        ),
        '*.ply',
        alphanum_sort=True
    )

    # all cloud_bin_* names
    fragment_names = [fn[:-4] for fn in fragment_names]

    poses = uio.read_log(osp.join(f'../benchmarks/{cfg.benchmarks}', scene_name, 'gt.log'))
    infos = uio.read_info_file(osp.join(f'../benchmarks/{cfg.benchmarks}', scene_name, 'gt.info'))

    register_results = []
    for pose in poses:
        i,j,num = pose.indices
        rr = RegisterResult(
            frag1_name=fragment_names[i],
            frag2_name=fragment_names[j],
            num_inliers=None,
            inlier_ratio=None,
            gt_flag=None,
        )
        register_results.append(rr)

    for k in tqdm.tqdm(range(len(register_results))):
        num_inliers, inlier_ratio, gt_flag,rs = register_fragment_pair(
            scene_name,
            seq_name,
            register_results[k].frag1_name,
            register_results[k].frag2_name,
            desc_type,
            poses,
            infos,
            pcloud_root,
            desc_root,
            inlier_thresh,
            k,
            cfg
        )

        rr, rre,rte,ir = rs

        register_results[k].num_inliers = num_inliers
        register_results[k].inlier_ratio = inlier_ratio
        register_results[k].gt_flag = gt_flag
        register_results[k].rr = rr
        register_results[k].rre = rre
        register_results[k].rte = rte
        register_results[k].ir = ir


    with open(osp.join(out_folder, out_filename + '.pkl'), 'wb') as fh:
        to_save = {
            'register_results': register_results,
            'scene_name': scene_name,
            'seq_name': seq_name,
            'desc_type': desc_type,
            'inlier_thresh': inlier_thresh,
        }
        pickle.dump(to_save, fh, protocol=pickle.HIGHEST_PROTOCOL)
    with open(osp.join(out_folder, out_filename + '.txt'), 'w') as fh:
        for k in register_results:
            fh.write('{} {} {} {:.8f} {} {} {} {} {}\n'.format(
                k.frag1_name,
                k.frag2_name,
                k.num_inliers,
                k.inlier_ratio,
                k.gt_flag,
                k.rr,
                k.rre,
                k.rte,
                k.ir)
            )
    return osp.join(out_folder, out_filename),len(poses)

def compute_metrics(
        match_paths,
        desc_type,
        inlier_thresh,
        out_root,
        scene_abbr_fn=None,
        scene_nums=None
):
    scenes = list()
    all_recalls = list()
    all_inliers = list()
    all_rr = list()
    all_rre = list()
    all_rte = list()
    all_ir = list()
    # scane result path
    for match_path,scene_num in zip(match_paths,scene_nums):
        with open(match_path + '.pkl', 'rb') as fh:
            saved = pickle.load(fh)
            register_results = saved['register_results']
            assert saved['inlier_thresh'] == inlier_thresh
        if scene_abbr_fn is not None:
            scenes.append(scene_abbr_fn(saved['scene_name']))
        else:
            scenes.append(saved['scene_name'])

        num_inliers = list()
        inlier_ratios = list()
        gt_flags = list()
        rrs = list()
        rres = list()
        rtes = list()
        irs = list()
        for rr in register_results:
            num_inliers.append(rr.num_inliers)
            inlier_ratios.append(rr.inlier_ratio)
            gt_flags.append(rr.gt_flag)
            rrs.append(rr.rr)
            rres.append(rr.rre)
            rtes.append(rr.rte)
            irs.append(rr.ir)
        num_inliers = np.asarray(num_inliers, dtype=np.int32)
        inlier_ratios = np.asarray(inlier_ratios, dtype=np.float32)
        gt_flags = np.asarray(gt_flags, dtype=np.int32)
        rrs = np.asarray(rrs,dtype=np.float32)
        rres = np.asarray(rres,dtype=np.float32)
        rtes = np.asarray(rtes,dtype=np.float32)
        irs = np.asarray(irs,dtype=np.float32)

        recalls = list()
        inliers = list()
        rr_recall = list()
        rre = list()
        rte = list()
        ir = list()
        for inlier_ratio_thresh in INLIER_RATIO_THRESHES:
            n_correct_matches = np.sum(inlier_ratios[gt_flags == 1] > inlier_ratio_thresh)
            recalls.append(float(n_correct_matches) / np.sum(gt_flags == 1))
            inliers.append(np.mean(num_inliers[gt_flags == 1]))

            rr_recall.append(np.sum(rrs))
            rre.append(np.sum(rres))
            rte.append(np.sum(rtes))
            ir.append(np.mean(irs))

        all_recalls.append(recalls)
        all_inliers.append(inliers)
        all_rr.append(rr_recall)
        all_rre.append(rre)
        all_rte.append(rte)
        all_ir.append(ir)


    out_path = osp.join(out_root, '{}-metrics-{:.2f}'.format(desc_type, inlier_thresh))
    with open(out_path + '.csv', 'w') as fh:
        header_str = 'SceneName'
        for inlier_ratio_thresh in INLIER_RATIO_THRESHES:
            header_str += ',Recall-{:.2f},AverageMatches-{:.2f}'.format(
                inlier_ratio_thresh, inlier_ratio_thresh)
        fh.write(header_str + '\n')

        for scene_name, recalls, inliers in zip(scenes, all_recalls, all_inliers):
            row_str = scene_name
            for recall, num_inlier in zip(recalls, inliers):
                row_str += ',{:.6f},{:.3f}'.format(recall, num_inlier)
            fh.write(row_str + '\n')

        avg_recalls = np.mean(np.asarray(all_recalls), axis=0).tolist()
        avg_inliers = np.mean(np.asarray(all_inliers), axis=0).tolist()

        avg_row_str = 'Average'
        for recall, num_inlier in zip(avg_recalls, avg_inliers):
            avg_row_str += ',{:.6f},{:.3f}'.format(recall, num_inlier)
        fh.write(avg_row_str + '\n')

    with open(out_path + '.pkl', 'wb') as fh:
        to_save = {
            'scenes': scenes,
            'recalls': all_recalls,
            'inliers': all_inliers,
            'rr': all_rr,
            'rre': all_rre,
            'rte': all_rte,
            'ir': all_ir,
            'threshes': INLIER_RATIO_THRESHES,
            'nums':np.sum(scene_nums)
        }
        pickle.dump(to_save, fh, protocol=pickle.HIGHEST_PROTOCOL)

    return out_path


def plot_recall_curve(desc_types, stat_paths, out_path):
    import matplotlib
    matplotlib.use('Agg')

    import matplotlib.pyplot as plt

    figure = plt.figure()
    for stat_path,desc_type in zip(stat_paths,desc_types):
        with open(stat_path + '.pkl', 'rb') as fh:
            saved = pickle.load(fh)
        threshes = np.asarray(saved['threshes'])
        all_recalls = np.asarray(saved['recalls'])
        avg_recalls = np.mean(all_recalls, axis=0)
        avg_std = np.std(all_recalls, axis=0)

        pair_num = saved["nums"]

        all_rr = np.asarray(saved["rr"])
        rr_pair = np.sum(all_rr,axis=0)
        avg_rr = rr_pair / pair_num
        all_rre = np.asarray(saved["rre"])
        avg_rre = np.sum(all_rre,axis=0) / rr_pair
        all_rte = np.asarray(saved["rte"])
        avg_rte = np.sum(all_rte,axis=0) / rr_pair
        all_ir = np.asarray(saved["ir"])
        avg_ir = np.mean(all_ir,axis=0)

        # plt.plot(threshes, avg_recalls * 100, linewidth=1)
        print(f"------- {desc_type} ---------")

        print(f"FMR:{avg_recalls}")
        print(f"STD:{avg_std}")

        print(f"Registration Recall:{avg_rr}")
        print(f"RRE:{avg_rre}")
        print(f"RTE:{avg_rte}")
        print(f"Inlier Ratio:{avg_ir}")

        print(f"------- {desc_type} ---------")

        plt.grid(True)
        plt.xlim(0, max(threshes))
        plt.xticks(np.arange(0, 6, dtype=np.float32) * max(threshes) / 5)
        plt.ylim(0, 100)
        plt.xlabel(r'$\tau_2$')
        plt.ylabel('Recall (%)')
        plt.legend(desc_types, loc='lower left')

    figure.savefig(out_path + '.pdf', bbox_inches='tight')


def evaluate(cfg):
    assert len(cfg.desc_types) == len(cfg.desc_roots)

    if cfg.mode == 'valid':
        scene_names = VALID_SCENE_NAMES
        scene_abbr_fn = None
    elif cfg.mode == 'test':
        scene_names = TEST_SCENE_NAMES
        scene_abbr_fn = lambda sn: TEST_SCENE_ABBR_NAMES[TEST_SCENE_NAMES.index(sn)]
    else:
        raise RuntimeError('[!] Mode is not supported.')

    # recurrent INLIER_THRESHES
    for inlier_thresh in INLIER_THRESHES:
        print('Start inlier_thresh {:.2f}m'.format(inlier_thresh))
        stat_paths = list()
        for desc_type, desc_root in zip(cfg.desc_types, cfg.desc_roots):

            print('  Start', desc_type)
            seq_name = 'seq-01'
            match_paths = list()
            scene_nums = list()

            for scene_name in scene_names:
                match_path, scene_num = run_scene_matching(
                    scene_name,
                    seq_name,
                    desc_type,
                    cfg.pcloud_root,
                    desc_root,
                    cfg.out_root,
                    inlier_thresh,
                    cfg,
                )
                match_paths.append(match_path)
                scene_nums.append(scene_num)

            stat_path = compute_metrics(
                match_paths,
                desc_type,
                inlier_thresh,
                cfg.out_root,
                scene_abbr_fn,
                scene_nums
            )
            stat_paths.append(stat_path)
        plot_recall_curve(
            cfg.desc_types,
            stat_paths,
            osp.join(cfg.out_root, 'recall-{:.2f}'.format(inlier_thresh))
        )

    print('All done.')


def parse_args():

    test_path = '/DISK/qwt/datasets/3dmatch/3DMatch_test'
    # out_path = "/DISK/qwt/desc/transformer/result"
    out_path = "/DISK/qwt/desc/transformer/good_result/Lo/nocat_198/result2/IMFNet_3DLoMatch_result"

    desc_path = "/DISK/qwt/desc/transformer/desc"
    desc_type = "IMFNet"

    desc_roots = [
        desc_path
    ]

    desc_types = [
        desc_type
    ]

    parser = argparse.ArgumentParser()
    parser.add_argument('--pcloud_root', default=test_path)
    parser.add_argument('--out_root', default=out_path)
    parser.add_argument('--desc_roots', nargs='+',default=desc_roots)
    parser.add_argument('--desc_types', nargs='+',default=desc_types)
    parser.add_argument('--mode', default='test')
    parser.add_argument('--voxel_size', default=0.025, type=float, help='voxel size to preprocess point cloud')
    parser.add_argument('--num_rand_keypoints', type=int, default=5000, help='random points')
    parser.add_argument('--keypoints',type=bool, default=True, help='wheather saving the keypoint')
    parser.add_argument('--benchmarks', default='3DLoMatch', help='3DMatch/3DLoMatch')

    return parser.parse_args()

if __name__ == '__main__':
    cfg = parse_args()
    evaluate(cfg)