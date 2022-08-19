"""
A collection of unrefactored functions.
"""
import warnings
warnings.filterwarnings("ignore")
import tqdm
import os
os.environ["CUDA_VISIBLE_DEVICES"]="0"
import sys
import numpy as np
import argparse
import logging
import open3d as o3d
import matplotlib.image as image
import math

ROOT_DIR = os.path.abspath('../')
if ROOT_DIR not in sys.path:
    sys.path.append(ROOT_DIR)


from lib.timer import Timer, AverageMeter
from util.misc import extract_features
from model import load_model
from util.file import ensure_dir, get_folder_list, get_file_list
from util.uio import process_image
import time

import torch

import MinkowskiEngine as ME

all_time = []
all_desc= []

ch = logging.StreamHandler(sys.stdout)
logging.getLogger().setLevel(logging.INFO)
logging.basicConfig(
    format='%(asctime)s %(message)s', datefmt='%m/%d %H:%M:%S', handlers=[ch])

o3d.utility.set_verbosity_level(o3d.utility.VerbosityLevel.Error)

# 抽取特征
def extract_features_batch(
        model,                              # model object
        config,                             # config object
        source_path,                        # testing path
        target_path,                        # descriptor path
        voxel_size,                         # voxel size, default 0.05
        device,                             # use GPU
):

  folders = get_folder_list(source_path)
  assert len(folders) > 0, f"Could not find 3DMatch folders under {source_path}"
  # logging.info(folders)
  for folder in folders:
      print(folder)
  # generate log file
  list_file = os.path.join(target_path, "list.txt")
  f = open(list_file, "w")
  timer, tmeter = Timer(), AverageMeter()
  num_feat = 0
  model.eval()

  for scene in folders:
    if 'evaluation' in scene:
      continue

    # getting all ply files in current scene
    files = get_file_list(scene+"/seq-01", ".ply")
    fo_base = os.path.basename(scene)
    f.write("%s %d\n" % (fo_base, len(files)))

    # dist seq path
    dist_seq_path = os.path.join(target_path,scene.split("/")[-1],"seq-01")
    if(not os.path.exists(dist_seq_path)):
        os.makedirs(dist_seq_path)

    all_desc.append(len(files))
    for i in tqdm.tqdm(range(len(files))):
      fi = files[i]
      # read PC
      pcd = o3d.io.read_point_cloud(fi)
      npz_filename = fi.split("/")[-1].replace(".ply",".npz")

      # read image
      image_file = fi.replace(".ply", "_0.png")
      suffix = ".png"
      if (not os.path.exists(image_file)):
          image_file = fi.replace(".ply", "_0.jpg")
          suffix = ".jpg"
      pc_image = image.imread(image_file)

      if (pc_image.shape[0] != config.image_H or pc_image.shape[1] != config.image_W):
          pc_image = process_image(image=pc_image, aim_H=config.image_H, aim_W=config.image_W)
      pc_image = np.transpose(pc_image, axes=(2, 0, 1))
      pc_image = np.expand_dims(pc_image,axis=0)

      start_time = time.time()
      xyz_down, feature = extract_features(
          model,
          xyz=np.array(pcd.points),
          rgb=None,
          normal=None,
          voxel_size=voxel_size,
          device=device,
          skip_check=True,
          image=pc_image
      )
      end_time = time.time() - start_time
      all_time.append(end_time)
      t = timer.toc()
      if i > 0:
        tmeter.update(t)
        num_feat += len(xyz_down)

      # saving descriptors to target_path
      np.savez_compressed(
          os.path.join(dist_seq_path, npz_filename),
          points=np.array(pcd.points),
          xyz=xyz_down,
          feature=feature.detach().cpu().numpy()
      )
      # print(f"Saving : {os.path.join(dist_seq_path, npz_filename)}")
      # print(f"------- Time : {end_time} s------- ")
      # if i % 20 == 0 and i > 0:
      #   logging.info(
      #       f'Average time: {tmeter.avg}, FPS: {num_feat / tmeter.sum}, time / feat: {tmeter.sum / num_feat}, '
      #   )

  f.close()
  os.remove(list_file)
  print("Descriptor Complete!")

if __name__ == '__main__':

  test_path = '/DISK/qwt/datasets/3dmatch/3DMatch_test'
  target_path = '/DISK/qwt/desc/transformer/desc'
  checkpoint_path = '/home/qwt/code/IMFNet-main/pretrain/3DMatch/3DMatch.pth'


  parser = argparse.ArgumentParser()
  parser.add_argument('--source', default=test_path, type=str, help='the path of 3DMatch testing')
  parser.add_argument('--target', default=target_path, type=str, help='the path of generating descriptor')
  parser.add_argument('-m', '--model', default=checkpoint_path, type=str, help='the path of checkpoints.pth')
  parser.add_argument('--voxel_size', default=0.05, type=float, help='voxel size to preprocess point cloud')
  parser.add_argument('--extract_features', default=True, action='store_true')
  parser.add_argument('--with_cuda', default=True, action='store_true')

  args = parser.parse_args()

  device = torch.device('cuda' if args.with_cuda else 'cpu')

  if args.extract_features:
      assert args.model is not None
      assert args.source is not None
      assert args.target is not None

      ensure_dir(args.target)
      checkpoint = torch.load(args.model)
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

      model = model.to(device)

      with torch.no_grad():

          extract_features_batch(
              model=model,
              config=config,
              source_path=args.source,
              target_path=args.target,
              voxel_size=config.voxel_size,
              device=device,
          )

          print(f"All Time:{np.sum(all_time)},AVG:{np.sum(all_time) / np.sum(all_desc)}")

