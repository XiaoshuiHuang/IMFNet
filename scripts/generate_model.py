"""
A collection of unrefactored functions.
"""
import os
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

from model import load_model
from util.file import ensure_dir, get_folder_list, get_file_list
import torch

ch = logging.StreamHandler(sys.stdout)
logging.getLogger().setLevel(logging.INFO)
logging.basicConfig(
    format='%(asctime)s %(message)s', datefmt='%m/%d %H:%M:%S', handlers=[ch])

o3d.utility.set_verbosity_level(o3d.utility.VerbosityLevel.Error)

def get_model(checkpoint_path):

    parser = argparse.ArgumentParser()

    parser.add_argument('--voxel_size', default=0.05, type=float, help='voxel size to preprocess point cloud')
    parser.add_argument('--extract_features', default=True, action='store_true')
    parser.add_argument('--evaluate_feature_match_recall', default=True, action='store_true')
    parser.add_argument('-m', '--model', default=checkpoint_path, type=str, help='checkpoints.pth的路径')
    parser.add_argument('--evaluate_registration', action='store_true', default=True,
                        help='The target directory must contain extracted features')
    parser.add_argument('--with_cuda', default=True, action='store_true')
    parser.add_argument('--num_rand_keypoints', type=int, default=5000, help='每个场景产生随机点的数量')

    args = parser.parse_args()

    if args.extract_features:
        assert args.model is not None

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

        return model,config

