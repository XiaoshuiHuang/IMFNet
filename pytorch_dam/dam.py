import cv2
import numpy as np
import torch
from pytorch_dam.base_dam import BaseDAM

class DAM(BaseDAM):
    def __init__(
            self,
            model,                              # model obj
            target_layer,                       # target layer obj
            use_cuda=False,                     # GPU
    ):
        super(DAM, self).__init__(model, target_layer, use_cuda)

    def get_cam_weights(self,
                        input_tensor,
                        target_category,
                        activations,
                        grads):

        return np.mean(grads, axis=1)