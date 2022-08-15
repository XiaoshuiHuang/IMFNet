# coding = utf-8
import torch


if __name__ == '__main__':

    checkpoint_path = "/home/qwt/code/IMFNet-main/pretrain/kitti.pth"
    x = torch.load(checkpoint_path)
    x_parameters = x["state_dict"]

    # for k,v in x_parameters.items():
    #     print(k)

    new_state_dict = dict()

    for k,v in x_parameters.items():
        if(k.__contains__("perceiver_io")):
            new_k = k.replace("perceiver_io","attention_fusion")
            new_state_dict[new_k] = v
            print(new_k)
        else:
            new_state_dict[k] = v
    x["state_dict"] = new_state_dict

    new_checkpoint_path = "/home/qwt/code/IMFNet-main/pretrain/Kitti_new.pth"
    torch.save(x,f=new_checkpoint_path)

