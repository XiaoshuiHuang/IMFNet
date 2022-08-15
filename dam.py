import argparse
import torch
import os
os.environ["CUDA_VISIBLE_DEVICES"]="1"
import warnings
warnings.filterwarnings("ignore")

from scripts.generate_model import get_model
from pytorch_dam import DAM
from pytorch_dam.utils.image import get_METensor,show_dam_on_pc_voxel_size


def get_args():
    checkpoint_path = "/home/qwt/code/IMFNet-main/pretrain/3DMatch/3DMatch.pth"

    parser = argparse.ArgumentParser()
    parser.add_argument('--use-cuda', action='store_true', default=True,help='Use NVIDIA GPU acceleration')
    parser.add_argument('--checkpoint', default=checkpoint_path,help='Model checkpoint.')
    parser.add_argument('--target', default=780,help='The target point index.')


    args = parser.parse_args()
    args.use_cuda = args.use_cuda and torch.cuda.is_available()
    if args.use_cuda:
        print('Using GPU for acceleration')
    else:
        print('Using CPU for computation')

    return args


if __name__ == '__main__':

    # model config
    args = get_args()

    # get model and config
    model,config = get_model(args.checkpoint)

    # the target point index
    target_point_index = args.target

    # the target layer
    target_layer = model.final

    dam = DAM(
        model=model,
        target_layer=target_layer,
        use_cuda=args.use_cuda
    )

    # point cloud
    ply_path = "files/cloud_bin_0.ply"
    image_path = "files/cloud_bin_0_0.png"

    print(f"Point cloud : {ply_path}")
    print(f"Image : {image_path}")
    print(f"Target Point Index: {target_point_index}")

    p_return_coords,p_pc,p_image,p_xyz,p_inds = get_METensor(
        pc_path = ply_path,
        image_path = image_path,
        image_H = config.image_H,
        image_W = config.image_W,
        voxel_size =  config.voxel_size,
        device = args.use_cuda
    )
    p_pc.requires_grad_()
    p_image.requires_grad_()

    # if(target_point_index == None):
    #     ply_path = "files/cloud_bin_1.ply"
    #     image_path = "files/cloud_bin_1_0.png"
    #     q_return_coords,q_pc,q_image,q_xyz,q_inds = get_METensor(
    #         pc_path = ply_path,
    #         image_path = image_path,
    #         image_H = config.image_H,
    #         image_W = config.image_W,
    #         voxel_size =  config.voxel_size,
    #         device = args.use_cuda
    #     )
    #     q_pc.requires_grad_()
    #     q_image.requires_grad_()

    input_tensor = (p_return_coords,p_pc,p_image)


    # feature map
    grayscale_dam, target_point_index = dam(
        input_tensor=input_tensor,
        target_category=target_point_index,
        target_layer=target_layer
    )

    # visualization the feature map
    show_dam_on_pc_voxel_size(
            dam=grayscale_dam,
            points=p_return_coords,
            target_point=target_point_index
    )
