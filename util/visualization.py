"""
A collection of unrefactored functions.
"""
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
import sys
import numpy as np
import matplotlib.image as image

ROOT_DIR = os.path.abspath('../')
if ROOT_DIR not in sys.path:
    sys.path.append(ROOT_DIR)
from scripts.benchmark_util import run_ransac

from util.misc import extract_features
from model import load_model
from util.uio import process_image
from util.pointcloud import make_open3d_point_cloud,make_open3d_feature_from_numpy
import open3d as o3d

import torch
import glob


import MinkowskiEngine as ME
def show():
    ply_path = "files/cloud_bin_0.ply"
    pc = o3d.io.read_point_cloud(ply_path)
    o3d.visualization.draw_geometries(
        [pc]
    )

def read_npz():
    npz_path = "files/cloud_bin_0.npz"
    data = np.load(npz_path)
    xyz=data["xyz"]
    feature=data["feature"]
    print(len(data["xyz"]))
    print(len(data["points"]))
    print(len(data["feature"]))
    print(data["xyz"])

def visualization_other():

    # read P and Q
    p_path = "select_files/cloud_bin_35.ply"
    q_path = "select_files/cloud_bin_53.ply"

    pc1 = o3d.io.read_point_cloud(p_path)
    pc2 = o3d.io.read_point_cloud(q_path)

    pc1.paint_uniform_color([0.4117647058823529, 0.3882352941176471, 0.9254901960784314])
    pc2.paint_uniform_color([0.9254901960784314, 0.9019607843137255, 0.2901960784313725])

    o3d.visualization.draw_geometries(
        [pc1, pc2]
    )

    T = [[ 0.7370,  0.5415, -0.4045,  0.1259],
        [-0.6319,  0.7644, -0.1282,  0.1724],
        [ 0.2398,  0.3501,  0.9055,  0.1371],
        [ 0.0000,  0.0000,  0.0000,  1.0000]]

    T = np.asarray(T)

    pc2.transform(T)

    o3d.visualization.draw_geometries(
        [pc1, pc2]
    )

def visualization_ground_truth():
    # read P and Q
    p_path = "files2/cloud_bin_0.ply"
    q_path = "files2/cloud_bin_1.ply"

    pc1 = o3d.io.read_point_cloud(p_path)
    pc2 = o3d.io.read_point_cloud(q_path)

    pc1.paint_uniform_color([0.4117647058823529, 0.3882352941176471, 0.9254901960784314])
    pc2.paint_uniform_color([0.9254901960784314, 0.9019607843137255, 0.2901960784313725])

    T = [
        [ 8.28189637e-01,-3.21528429e-01,4.59044120e-01,-2.00031198e-01],
        [ 2.37171821e-01,9.43178735e-01,2.32734064e-01,5.16811398e-02],
        [ -5.07789707e-01,-8.38762590e-02,8.57387693e-01,-4.06217836e-02],
        [  0.00000000e+00, 0.00000000e+00,0.00000000e+00,1.00000000e+00],
    ]

    T = np.asarray(T)

    pc2.transform(T)

    o3d.visualization.draw_geometries(
        [pc1, pc2]
    )

def visualization_ours(voxel_size=0.025):

  # read P and Q
  p_path = "files/cloud_bin_0.ply"
  q_path = "files/cloud_bin_1.ply"

  pc1 = o3d.io.read_point_cloud(p_path)
  pc2 = o3d.io.read_point_cloud(q_path)

  p_xyz = np.asarray(pc1.points)
  q_xyz = np.asarray(pc2.points)

  # load the model
  checkpoint_path = "../outputs/checkpoint_epoch_198_0.985.pth"
  checkpoint = torch.load(checkpoint_path)
  config = checkpoint['config']

  num_feats = 1
  Model = load_model(config.model)
  model = Model(
      num_feats,
      config.model_n_out,
      bn_momentum=0.05,
      normalize_feature=config.normalize_feature,
      conv1_kernel_size=config.conv1_kernel_size,
      D=3,
      config=config
  )
  model.load_state_dict(checkpoint['state_dict'])
  model.eval()
  device = torch.device("cuda")
  model = model.to(device)

  # read p_image and q_image
  p_image_path = "files/cloud_bin_0_0.png"
  q_image_path = "files/cloud_bin_1_0.png"

  p_image = image.imread(p_image_path)
  if (p_image.shape[0] != config.image_H or p_image.shape[1] != config.image_W):
      p_image = process_image(image=p_image, aim_H=config.image_H, aim_W=config.image_W)
  p_image = np.transpose(p_image, axes=(2, 0, 1))
  p_image = np.expand_dims(p_image, axis=0)

  q_image = image.imread(q_image_path)
  if (q_image.shape[0] != config.image_H or q_image.shape[1] != config.image_W):
      q_image = process_image(image=q_image, aim_H=config.image_H, aim_W=config.image_W)
  q_image = np.transpose(q_image, axes=(2, 0, 1))
  q_image = np.expand_dims(q_image, axis=0)

  # generate f_p and f_q
  p_xyz_down, p_feature = extract_features(
      model,
      xyz=p_xyz,
      rgb=None,
      normal=None,
      voxel_size=voxel_size,
      device=device,
      skip_check=True,
      image=p_image
  )

  q_xyz_down, q_feature = extract_features(
      model,
      xyz=q_xyz,
      rgb=None,
      normal=None,
      voxel_size=voxel_size,
      device=device,
      skip_check=True,
      image=q_image
  )

  # get the evaluation metrix
  p_xyz_down = make_open3d_point_cloud(p_xyz_down)
  q_xyz_down = make_open3d_point_cloud(q_xyz_down)

  p_feature = p_feature.cpu().detach().numpy()
  p_feature = make_open3d_feature_from_numpy(p_feature)
  q_feature = q_feature.cpu().detach().numpy()
  q_feature = make_open3d_feature_from_numpy(q_feature)
  T = run_ransac(
      p_xyz_down,
      q_xyz_down,
      p_feature,
      q_feature,
      voxel_size
  )

  pc1.paint_uniform_color([0.4117647058823529,0.3882352941176471,0.9254901960784314])
  pc2.paint_uniform_color([0.9254901960784314,0.9019607843137255,0.2901960784313725])
  o3d.visualization.draw_geometries(
      [pc1,pc2]
  )
  pc1.transform(T)
  o3d.visualization.draw_geometries(
      [pc1,pc2]
  )

def spinnet_desc():

    desc_path = "/DISK/qwt/desc/test_registration/SpinNet_desc"
    keypoint_path = "/DISK/qwt/datasets/3dmatch/SpinNet_pkl/3DMatch/merge_zip/3DMatch/keypoints"

    desc_output_path = "/DISK/qwt/desc/test_registration/SpinNet"
    scenes = [scene.split("/")[-1]for scene in glob.glob(desc_path+"/*")]
    for scene in scenes:
        desc_output_path_ = os.path.join(desc_output_path,scene,"seq-01")
        if(not os.path.exists(desc_output_path_)):
            os.makedirs(desc_output_path_)
        desc_path_ = os.path.join(desc_path,scene,"*.npy")
        descs = [desc.split("/")[-1] for desc in glob.glob(desc_path_)]

        for desc in descs:

            id = desc.split(".")[0]
            desc_keypoiont_output_path = os.path.join(desc_output_path_,id+".npz")

            keypoint = desc.replace(".desc.SpinNet.bin","_keypts")
            keypoint = os.path.join(keypoint_path,scene,keypoint)
            xyz = np.load(keypoint)

            desc = os.path.join(desc_path,scene,desc)
            feature = np.load(desc)

            np.savez(
                file=desc_keypoiont_output_path,
                xyz=xyz,
                feature=feature
            )

            print(f"Saving : {desc_keypoiont_output_path}")

from util.file import ensure_dir, get_folder_list, get_file_list
from scripts.evluation_3dmatch_test import read_log

def visualization_3DMatch(voxel_size=0.025):
      # mano
      test_path = "/DISK/qwt/datasets/Ours_train_0_01/test"
      # test_path = "/DISK/qwt/datasets/3dmatch/3DLoMatch_test"

      FCGF = "/DISK/qwt/desc/test_registration/FCGF"
      PointImageNet = "/DISK/qwt/desc/test_registration/PointImageNet"
      Predator = "/DISK/qwt/desc/test_registration/Predator"
      SpinNet = "/DISK/qwt/desc/test_registration/SpinNet"

      # result_path = "/DISK/qwt/desc/test_registration/result"
      # result_path = "/DISK/qwt/desc/test_registration/result_Lo"
      result_path = "/DISK/qwt/desc/test_registration/result_noLo"
      # select_path = "/DISK/qwt/desc/test_registration/result_select"
      # select_path = "/DISK/qwt/desc/test_registration/result_select_Lo"
      select_path = "/DISK/qwt/desc/test_registration/result_select_noLo"

      scenes = [scene.split("/")[-1] for scene in get_folder_list(test_path)]
      for scene in scenes:
          poses = read_log(os.path.join(test_path, scene, "seq-01", 'gt.log'))
          Predator_path = os.path.join(Predator,scene,"seq-01","*_p.npz")
          Predator_descs = [desc.split("/")[-1] for desc in glob.glob(Predator_path)]
          for desc in Predator_descs:

              # p_id,q_id
              p_id = int(desc.split("-")[0].split("_")[-1])
              q_id = int(desc.split("-")[1].split("_")[2])

              print(f"---- {scene} cloud_bin_{p_id}-cloud_bin_{q_id} ----")

              file_name = f"{scene}_cloud_bin_{p_id}-cloud_bin_{q_id}.txt"
              file_path = os.path.join(result_path,file_name)
              if(os.path.exists(file_path)):
                  print("Skip!")
                  continue

              pose_id = -1
              for pid, pose in enumerate(poses):
                  # 判断当前的P索引和Q索引是否是与给出的点云P和点云Q下标一致
                  if pose.indices[0] == p_id and pose.indices[1] == q_id:
                      pose_id = pid
                      break
              if(pose_id == -1):
                   continue
              # P
              Predator_desc_p_path = os.path.join(Predator, scene, "seq-01", desc)
              FCGF_desc_p_path = os.path.join(FCGF,scene,"seq-01",f"cloud_bin_{p_id}.npz")
              PointImageNet_desc_p_path = os.path.join(PointImageNet,scene,"seq-01",f"cloud_bin_{p_id}.npz")
              SpinNet_desc_p_path = os.path.join(SpinNet,scene,"seq-01",f"cloud_bin_{p_id}.npz")

              Predator_p = np.load(Predator_desc_p_path)
              FCGF_p = np.load(FCGF_desc_p_path)
              PointImageNet_p = np.load(PointImageNet_desc_p_path)
              SpinNet_p = np.load(SpinNet_desc_p_path)

              # xyz
              Predator_p_xyz = make_open3d_point_cloud(Predator_p["xyz"])
              FCGF_p_xyz = make_open3d_point_cloud(FCGF_p["xyz"])
              PointImageNet_p_xyz = make_open3d_point_cloud(PointImageNet_p["xyz"])
              SpinNet_p_xyz = make_open3d_point_cloud(SpinNet_p["xyz"])

              # feature
              Predator_p_feature = make_open3d_feature_from_numpy(Predator_p["feature"])
              FCGF_p_feature = make_open3d_feature_from_numpy(FCGF_p["feature"])
              PointImageNet_p_feature = make_open3d_feature_from_numpy(PointImageNet_p["feature"])
              SpinNet_p_feature = make_open3d_feature_from_numpy(SpinNet_p["feature"])

              # Q
              Predator_desc_q_path = os.path.join(Predator, scene, "seq-01",desc.replace("_p.npz","_q.npz"))
              FCGF_desc_q_path = os.path.join(FCGF,scene,"seq-01",f"cloud_bin_{q_id}.npz")
              PointImageNet_desc_q_path = os.path.join(PointImageNet,scene,"seq-01",f"cloud_bin_{q_id}.npz")
              SpinNet_desc_q_path = os.path.join(SpinNet,scene,"seq-01",f"cloud_bin_{q_id}.npz")

              Predator_q = np.load(Predator_desc_q_path)
              FCGF_q = np.load(FCGF_desc_q_path)
              PointImageNet_q = np.load(PointImageNet_desc_q_path)
              SpinNet_q = np.load(SpinNet_desc_q_path)

              # xyz
              Predator_q_xyz = make_open3d_point_cloud(Predator_q["xyz"])
              FCGF_q_xyz = make_open3d_point_cloud(FCGF_q["xyz"])
              PointImageNet_q_xyz = make_open3d_point_cloud(PointImageNet_q["xyz"])
              SpinNet_q_xyz = make_open3d_point_cloud(SpinNet_q["xyz"])

              # feature
              Predator_q_feature = make_open3d_feature_from_numpy(Predator_q["feature"])
              FCGF_q_feature = make_open3d_feature_from_numpy(FCGF_q["feature"])
              PointImageNet_q_feature = make_open3d_feature_from_numpy(PointImageNet_q["feature"])
              SpinNet_q_feature = make_open3d_feature_from_numpy(SpinNet_q["feature"])

              # Ransac -> T
              Predator_T_ = run_ransac(
                              Predator_q_xyz,
                              Predator_p_xyz,
                              Predator_q_feature,
                              Predator_p_feature,
                              voxel_size
                          )
              Predator_T = torch.from_numpy(Predator_T_.astype(np.float32))

              FCGF_T_ = run_ransac(
                              FCGF_q_xyz,
                              FCGF_p_xyz,
                              FCGF_q_feature,
                              FCGF_p_feature,
                              voxel_size
                          )
              FCGF_T = torch.from_numpy(FCGF_T_.astype(np.float32))
              PointImageNet_T_ = run_ransac(
                              PointImageNet_q_xyz,
                              PointImageNet_p_xyz,
                              PointImageNet_q_feature,
                              PointImageNet_p_feature,
                              voxel_size
                          )
              PointImageNet_T = torch.from_numpy(PointImageNet_T_.astype(np.float32))
              SpinNet_T_ = run_ransac(
                              SpinNet_q_xyz,
                              SpinNet_p_xyz,
                              SpinNet_q_feature,
                              SpinNet_p_feature,
                              voxel_size
                          )
              SpinNet_T = torch.from_numpy(SpinNet_T_.astype(np.float32))
              # rte rre
              T_gth = poses[pose_id].transformation

              Predator_rte = np.linalg.norm(Predator_T[:3, 3] - T_gth[:3, 3])
              Predator_rre = np.arccos((np.trace(Predator_T[:3, :3].t() @ T_gth[:3, :3]) - 1) / 2)

              FCGF_rte = np.linalg.norm(FCGF_T[:3, 3] - T_gth[:3, 3])
              FCGF_rre = np.arccos((np.trace(FCGF_T[:3, :3].t() @ T_gth[:3, :3]) - 1) / 2)

              PointImageNet_rte = np.linalg.norm(PointImageNet_T[:3, 3] - T_gth[:3, 3])
              PointImageNet_rre = np.arccos((np.trace(PointImageNet_T[:3, :3].t() @ T_gth[:3, :3]) - 1) / 2)

              SpinNet_rte = np.linalg.norm(SpinNet_T[:3, 3] - T_gth[:3, 3])
              SpinNet_rre = np.arccos((np.trace(SpinNet_T[:3, :3].t() @ T_gth[:3, :3]) - 1) / 2)

              fige = True
              rte_thresh = 0.3
              rre_thresh = np.pi / 180 * 15
              if(PointImageNet_rte < rte_thresh and not np.isnan(PointImageNet_rre) and PointImageNet_rre < rre_thresh):
                  file_name = f"{scene}_cloud_bin_{p_id}-cloud_bin_{q_id}.txt"
                  file_path = os.path.join(result_path, file_name)
                  with open(file=file_path, mode="w") as f:
                      f.write(f"PointImageNet---rte:{PointImageNet_rte},rre:{PointImageNet_rre},T:\n{PointImageNet_T_}")
                      f.write(f"Ground Truth,T:\n{T_gth}\n")
                  print(f"Select_Saving : {file_path}")
              # select
              if(Predator_rte < rte_thresh and not np.isnan(Predator_rre) and Predator_rre < rre_thresh ):
                  fige = False
              if(FCGF_rte < rte_thresh and not np.isnan(FCGF_rre) and FCGF_rre < rre_thresh ):
                  fige = False
              if (SpinNet_rte < rte_thresh and not np.isnan(SpinNet_rre) and SpinNet_rre < rre_thresh):
                  fige = False
              if (PointImageNet_rte > rte_thresh or np.isnan(PointImageNet_rre) or PointImageNet_rre > rre_thresh):
                  fige = False
              if(fige):
                  print("******************************************************************************")
                  print(f"Predator--rte:{Predator_rte},rre:{Predator_rre},T:\n{Predator_T_}",end="")
                  print(f"FCGF--rte:{FCGF_rte},rre:{FCGF_rre},T:\n{FCGF_T_}",end="")
                  print(f"SpinNet--rte:{SpinNet_rte},rre:{SpinNet_rre},T:\n{SpinNet_T_}",end="")
                  print(f"PointImageNet--rte:{PointImageNet_rte},rre:{PointImageNet_rre},T:\n{PointImageNet_T_}")

                  file_name = f"{scene}_cloud_bin_{p_id}-cloud_bin_{q_id}.txt"
                  file_path = os.path.join(select_path, file_name)
                  with open(file=file_path, mode="w") as f:
                      f.write(f"Predator---rte:{Predator_rte},rre:{Predator_rre},T:\n{Predator_T_}\n")
                      f.write(f"FCGF---rte:{FCGF_rte},rre:{FCGF_rre},T:\n{FCGF_T_}\n")
                      f.write(f"PointImageNet---rte:{PointImageNet_rte},rre:{PointImageNet_rre},T:\n{PointImageNet_T_}\n")
                      f.write(f"SpinNet---rte:{SpinNet_rte},rre:{SpinNet_rre},T:\n{SpinNet_T_}\n")
                      f.write(f"Ground Truth,T:\n{T_gth}")
                  print(f"Select_Saving : {file_path}")

                  print("******************************************************************************")


def visualization_Kitti(voxel_size=0.05):
    # mano
    seqs = [8]

    FCGF = "/DISK/qwt/desc/test_registration_kitti/FCGF"
    PointImageNet = "/DISK/qwt/desc/test_registration_kitti/PointImageNet"
    Predator = "/DISK/qwt/desc/test_registration_kitti/Predator"
    SpinNet = "/DISK/qwt/desc/test_registration_kitti/SpinNet"

    select_path = "/DISK/qwt/desc/test_registration_kitti/select_result"
    result_path = "/DISK/qwt/desc/test_registration_kitti/result"

    best_rre_id = -1
    best_rte_id = -1
    min_rre = 100000000
    min_rte = 100000000
    Predator_rre_T = None
    FCGF_rre_T = None
    PointImageNet_rre_T = None
    SpinNet_rre_T = None

    Predator_rtte_T = None
    FCGF_rte_T = None
    PointImageNet_rte_T = None
    SpinNet_rte_T = None

    for seq in seqs:

        PointImageNet_path = os.path.join(PointImageNet, f"{seq}","*_p.npz")
        PointImageNet_descs = [desc.split("/")[-1].split("_")[1] for desc in glob.glob(PointImageNet_path)]
        for index,desc in enumerate(PointImageNet_descs):

            # p_id,q_id
            p_id = int(desc.split("-")[0])
            q_id = int(desc.split("-")[1])

            print(f"---- {seq} cloud_bin_{p_id}-cloud_bin_{q_id},{index}-{len(PointImageNet_descs)} ----")

            file_name = f"{seq}_{p_id}-{q_id}_p.txt"
            file_path = os.path.join(select_path, file_name)
            if (os.path.exists(file_path)):
                print("Skip!")
                continue

            # P
            p_name = f"{seq}_{p_id}-{q_id}_p.npz"
            Predator_desc_p_path = os.path.join(Predator, f"{seq}", p_name)
            FCGF_desc_p_path = os.path.join(FCGF, f"{seq}", p_name)
            PointImageNet_desc_p_path = os.path.join(PointImageNet,f"{seq}", p_name)
            SpinNet_desc_p_path = os.path.join(SpinNet, f"{seq}", p_name)

            Predator_p = np.load(Predator_desc_p_path)
            FCGF_p = np.load(FCGF_desc_p_path)
            PointImageNet_p = np.load(PointImageNet_desc_p_path)
            SpinNet_p = np.load(SpinNet_desc_p_path)

            # xyz
            Predator_p_xyz = make_open3d_point_cloud(Predator_p["xyz"])
            FCGF_p_xyz = make_open3d_point_cloud(FCGF_p["xyz"])
            PointImageNet_p_xyz = make_open3d_point_cloud(PointImageNet_p["xyz"])
            SpinNet_p_xyz = make_open3d_point_cloud(SpinNet_p["xyz"])

            # feature
            Predator_p_feature = make_open3d_feature_from_numpy(Predator_p["feature"])
            FCGF_p_feature = make_open3d_feature_from_numpy(FCGF_p["feature"])
            PointImageNet_p_feature = make_open3d_feature_from_numpy(PointImageNet_p["feature"])
            SpinNet_p_feature = make_open3d_feature_from_numpy(SpinNet_p["feature"])

            # Q
            q_name = f"{seq}_{p_id}-{q_id}_q.npz"
            Predator_desc_q_path = os.path.join(Predator,f"{seq}",q_name)
            FCGF_desc_q_path = os.path.join(FCGF,f"{seq}",q_name)
            PointImageNet_desc_q_path = os.path.join(PointImageNet,f"{seq}",q_name)
            SpinNet_desc_q_path = os.path.join(SpinNet,f"{seq}",q_name)

            Predator_q = np.load(Predator_desc_q_path)
            FCGF_q = np.load(FCGF_desc_q_path)
            PointImageNet_q = np.load(PointImageNet_desc_q_path)
            SpinNet_q = np.load(SpinNet_desc_q_path)

            # xyz
            Predator_q_xyz = make_open3d_point_cloud(Predator_q["xyz"])
            FCGF_q_xyz = make_open3d_point_cloud(FCGF_q["xyz"])
            PointImageNet_q_xyz = make_open3d_point_cloud(PointImageNet_q["xyz"])
            SpinNet_q_xyz = make_open3d_point_cloud(SpinNet_q["xyz"])

            # feature
            Predator_q_feature = make_open3d_feature_from_numpy(Predator_q["feature"])
            FCGF_q_feature = make_open3d_feature_from_numpy(FCGF_q["feature"])
            PointImageNet_q_feature = make_open3d_feature_from_numpy(PointImageNet_q["feature"])
            SpinNet_q_feature = make_open3d_feature_from_numpy(SpinNet_q["feature"])

            # Ransac -> T
            Predator_T_ = run_ransac(
                Predator_p_xyz,
                Predator_q_xyz,
                Predator_p_feature,
                Predator_q_feature,
                voxel_size
            )
            Predator_T = torch.from_numpy(Predator_T_.astype(np.float32))

            FCGF_T_ = run_ransac(
                FCGF_p_xyz,
                FCGF_q_xyz,
                FCGF_p_feature,
                FCGF_q_feature,
                voxel_size
            )
            FCGF_T = torch.from_numpy(FCGF_T_.astype(np.float32))
            PointImageNet_T_ = run_ransac(
                PointImageNet_p_xyz,
                PointImageNet_q_xyz,
                PointImageNet_p_feature,
                PointImageNet_q_feature,
                voxel_size
            )
            PointImageNet_T = torch.from_numpy(PointImageNet_T_.astype(np.float32))
            SpinNet_T_ = run_ransac(
                SpinNet_p_xyz,
                SpinNet_q_xyz,
                SpinNet_p_feature,
                SpinNet_q_feature,
                voxel_size
            )
            SpinNet_T = torch.from_numpy(SpinNet_T_.astype(np.float32))
            # rte rre
            T_gth = np.asarray(PointImageNet_p["T_gth"])

            Predator_rte = np.linalg.norm(Predator_T[:3, 3] - T_gth[:3, 3])
            Predator_rre = np.arccos((np.trace(Predator_T[:3, :3].t() @ T_gth[:3, :3]) - 1) / 2)

            FCGF_rte = np.linalg.norm(FCGF_T[:3, 3] - T_gth[:3, 3])
            FCGF_rre = np.arccos((np.trace(FCGF_T[:3, :3].t() @ T_gth[:3, :3]) - 1) / 2)

            PointImageNet_rte = np.linalg.norm(PointImageNet_T[:3, 3] - T_gth[:3, 3])
            PointImageNet_rre = np.arccos((np.trace(PointImageNet_T[:3, :3].t() @ T_gth[:3, :3]) - 1) / 2)

            SpinNet_rte = np.linalg.norm(SpinNet_T[:3, 3] - T_gth[:3, 3])
            SpinNet_rre = np.arccos((np.trace(SpinNet_T[:3, :3].t() @ T_gth[:3, :3]) - 1) / 2)

            if(not np.isnan(PointImageNet_rre) and not np.isnan(PointImageNet_rte)):
                if(PointImageNet_rre < min_rre):
                    min_rre = PointImageNet_rre
                    best_rre_id = [p_id,q_id]
                    Predator_rre_T = Predator_T_
                    FCGF_rre_T = FCGF_T_
                    PointImageNet_rre_T = PointImageNet_T_
                    SpinNet_rre_T = SpinNet_T_
                if(PointImageNet_rte < min_rte):
                    min_rte = PointImageNet_rte
                    best_rte_id = [p_id,q_id]
                    Predator_rte_T = Predator_T_
                    FCGF_rte_T = FCGF_T_
                    PointImageNet_rte_T = PointImageNet_T_
                    SpinNet_rte_T = SpinNet_T_

            # print(f"Predator,rre:{Predator_rre},rte:{Predator_rte}")
            # print(f"FCGF,rre:{FCGF_rre},rte:{FCGF_rte}")
            # print(f"PointImageNet,rre:{PointImageNet_rre},rte:{PointImageNet_rte}")
            # print(f"SpinNet,rre:{SpinNet_rre},rte:{SpinNet_rte}")
            fige = True
            rte_thresh = 0.2
            rre_thresh = np.pi / 180 * 5
            # select
            if (Predator_rte < rte_thresh and not np.isnan(Predator_rre) and Predator_rre < rre_thresh):
                fige = False
            if (FCGF_rte < rte_thresh and not np.isnan(FCGF_rre) and FCGF_rre < rre_thresh):
                fige = False
            if (SpinNet_rte < rte_thresh and not np.isnan(SpinNet_rre) and SpinNet_rre < rre_thresh):
                fige = False
            if (PointImageNet_rte > rte_thresh or np.isnan(PointImageNet_rre) or PointImageNet_rre > rre_thresh):
                fige = False
            if (fige):
                print("******************************************************************************")
                print(f"Predator--rte:{Predator_rte},rre:{Predator_rre},T:\n{Predator_T_}")
                print(f"FCGF--rte:{FCGF_rte},rre:{FCGF_rre},T:\n{FCGF_T_}")
                print(f"SpinNet--rte:{SpinNet_rte},rre:{SpinNet_rre},T:\n{SpinNet_T_}")
                print(f"PointImageNet--rte:{PointImageNet_rte},rre:{PointImageNet_rre},T:\n{PointImageNet_T_}")

                file_path = os.path.join(select_path, file_name)
                with open(file=file_path, mode="w") as f:
                    f.write(f"Predator---rte:{Predator_rte},rre:{Predator_rre},T:\n{Predator_T_}\n")
                    f.write(f"FCGF---rte:{FCGF_rte},rre:{FCGF_rre},T:\n{FCGF_T_}\n")
                    f.write(f"PointImageNet---rte:{PointImageNet_rte},rre:{PointImageNet_rre},T:\n{PointImageNet_T_}\n")
                    f.write(f"SpinNet---rte:{SpinNet_rte},rre:{SpinNet_rre},T:\n{PointImageNet_T_}\n")
                    f.write(f"Ground Truth,T:\n{T_gth}\n")
                print(f"Select_Saving : {file_path}")

                print("******************************************************************************")

            if(PointImageNet_rte < Predator_rte and PointImageNet_rte < FCGF_rte and PointImageNet_rte < SpinNet_rte and \
                PointImageNet_rre <Predator_rre and PointImageNet_rre < FCGF_rre and PointImageNet_rre < SpinNet_rre):
                print(f"Predator--rte:{Predator_rte},rre:{Predator_rre},T:\n{Predator_T_}")
                print(f"FCGF--rte:{FCGF_rte},rre:{FCGF_rre},T:\n{FCGF_T_}")
                print(f"SpinNet--rte:{SpinNet_rte},rre:{SpinNet_rre},T:\n{SpinNet_T_}")
                print(f"PointImageNet--rte:{PointImageNet_rte},rre:{PointImageNet_rre},T:\n{PointImageNet_T_}")
                file_path = os.path.join(select_path, file_name)
                with open(file=file_path, mode="w") as f:
                    f.write(f"Predator---rte:{Predator_rte},rre:{Predator_rre},T:\n{Predator_T_}\n")
                    f.write(f"FCGF---rte:{FCGF_rte},rre:{FCGF_rre},T:\n{FCGF_T_}\n")
                    f.write(f"PointImageNet---rte:{PointImageNet_rte},rre:\n{PointImageNet_rre},T:{PointImageNet_T_}\n")
                    f.write(f"SpinNet---rte:{SpinNet_rte},rre:{SpinNet_rre},T:\n{PointImageNet_T_}\n")
                    f.write(f"Ground Truth,T:\n{T_gth}\n")
                print(f"Select_Saving : {file_path}")

                print("******************************************************************************")

            file_path = os.path.join(result_path, file_name)
            with open(file=file_path, mode="w") as f:
                f.write(f"Predator---rte:{Predator_rte},rre:{Predator_rre},T:\n{Predator_T_}\n")
                f.write(f"FCGF---rte:{FCGF_rte},rre:{FCGF_rre},T:\n{FCGF_T_}\n")
                f.write(f"PointImageNet---rte:{PointImageNet_rte},rre:\n{PointImageNet_rre},T:{PointImageNet_T_}\n")
                f.write(f"SpinNet---rte:{SpinNet_rte},rre:{SpinNet_rre},T:\n{PointImageNet_T_}\n")
                f.write(f"Ground Truth,T:\n{T_gth}\n")
            print(f"Select_Saving : {file_path}")
            print("--------------------------------------------------------------------")
    print(f"min_rre:{min_rre},best_rre_id:{best_rre_id},min_rte:{min_rte},best_rte_id:{best_rte_id}")
    print("rre:")
    print(Predator_rre_T)
    print(FCGF_rre_T)
    print(PointImageNet_rre_T)
    print(SpinNet_rre_T)

    print("rte")
    print(Predator_rte_T)
    print(FCGF_rte_T)
    print(PointImageNet_rte_T)
    print(SpinNet_rte_T)
if __name__ == '__main__':
    # read_npz()
    # visualization_ours()
    # visualization_ground_truth()
    # spinnet_desc()
    visualization_3DMatch()
    # visualization_Kitti()