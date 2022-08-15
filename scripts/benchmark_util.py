import open3d as o3d
import os
import logging
import numpy as np

from util.trajectory import CameraPose
from util.pointcloud import compute_overlap_ratio, \
    make_open3d_point_cloud, make_open3d_feature_from_numpy

# open3d == 0.12.0.0
"""
RANSAC pose estimation with two checkers
We follow D3Feat,PREDATOR to set ransac_n = 3 for 3DMatch and ransac_n = 4 for KITTI. 
For 3DMatch dataset, we observe significant improvement after changing ransac_n from 4 to 3.
"""
def run_ransac(xyz0, xyz1, feat0, feat1, voxel_size,ransac_n=4):
  distance_threshold = voxel_size * 1.5
  result_ransac = o3d.pipelines.registration.registration_ransac_based_on_feature_matching(
      source=xyz0,
      target=xyz1,
      source_feature=feat0,
      target_feature=feat1,
      max_correspondence_distance=distance_threshold,
      estimation_method=o3d.pipelines.registration.TransformationEstimationPointToPoint(False),
      ransac_n=ransac_n,
      checkers=[
          o3d.pipelines.registration.CorrespondenceCheckerBasedOnEdgeLength(0.9),
          o3d.pipelines.registration.CorrespondenceCheckerBasedOnDistance(distance_threshold)
      ],
      # criteria=o3d.pipelines.registration.RANSACConvergenceCriteria(4000000, 500),
      criteria=o3d.pipelines.registration.RANSACConvergenceCriteria(50000, 1000),
      mutual_filter=False
  )
  return result_ransac.transformation

# open3d == 0.9.0.0
# def run_ransac(xyz0, xyz1, feat0, feat1, voxel_size,ransac_n=4):
#   distance_threshold = voxel_size * 1.5
#   result_ransac = o3d.registration.registration_ransac_based_on_feature_matching(
#       xyz0, xyz1, feat0, feat1, distance_threshold,
#       o3d.registration.TransformationEstimationPointToPoint(False), ransac_n, [
#           o3d.registration.CorrespondenceCheckerBasedOnEdgeLength(0.9),
#           o3d.registration.CorrespondenceCheckerBasedOnDistance(distance_threshold)
#       ],
#     # o3d.registration.RANSACConvergenceCriteria(4000000, 500)
#     o3d.registration.RANSACConvergenceCriteria(50000, 1000)
#   )
#   return result_ransac.transformation

def gather_results(results):
  traj = []
  for r in results:
    success = r[0]
    if success:
      traj.append(CameraPose([r[1], r[2], r[3]], r[4]))
  return traj


def gen_matching_pair(pts_num):
  matching_pairs = []
  for i in range(pts_num):
    for j in range(i + 1, pts_num):
      matching_pairs.append([i, j, pts_num])
  return matching_pairs


def read_data(feature_path, name):
  data = np.load(os.path.join(feature_path, name + ".npz"))
  xyz = make_open3d_point_cloud(data['xyz'])
  feat = make_open3d_feature_from_numpy(data['feature'])
  return data['points'], xyz, feat


def do_single_pair_matching(feature_path, set_name, m, voxel_size):
  i, j, s = m
  name_i = "%s_%03d" % (set_name, i)
  name_j = "%s_%03d" % (set_name, j)
  logging.info("matching %s %s" % (name_i, name_j))
  points_i, xyz_i, feat_i = read_data(feature_path, name_i)
  points_j, xyz_j, feat_j = read_data(feature_path, name_j)
  if len(xyz_i.points) < len(xyz_j.points):
    trans = run_ransac(xyz_i, xyz_j, feat_i, feat_j, voxel_size)
  else:
    trans = run_ransac(xyz_j, xyz_i, feat_j, feat_i, voxel_size)
    trans = np.linalg.inv(trans)
  ratio = compute_overlap_ratio(xyz_i, xyz_j, trans, voxel_size)
  logging.info(f"{ratio}")
  if ratio > 0.3:
    return [True, i, j, s, np.linalg.inv(trans)]
  else:
    return [False, i, j, s, np.identity(4)]
