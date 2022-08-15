# coding = utf-8
import numpy as np
import open3d as o3d
from pytorch_dam.activations_and_gradients import ActivationsAndGradients


class BaseDAM:
    def __init__(
            self,
            model,
            target_layer,
            use_cuda=False,
    ):
        self.model = model.eval()
        self.target_layer = target_layer

        if use_cuda:
            self.model = model.cuda()
        # get the feature map and gradient
        self.activations_and_grads = ActivationsAndGradients(
            self.model,
            target_layer
        )

    def forward(self, input_img):
        return self.model(input_img)

    def get_cam_weights(self,
                        input_tensor,
                        target_category,
                        activations,
                        grads):
        raise Exception("Not Implemented")


    def get_loss(
            self,
            p_feature,
            p_xyz,
            target_category
    ):
        target_point_feature = p_feature[target_category,:]
        target_point_coord = p_xyz[target_category,:]
        return target_point_feature,target_point_coord

    def get_feature_inds(
            self,
            source,
            target,
            k=30
    ):

        kdtree = o3d.geometry.KDTreeFlann(np.asarray(target.T, dtype=np.float64))
        source = np.asarray(source, dtype=np.float64)
        match_inds = []
        for i,feature in enumerate(source):
            [_,idx,_] = kdtree.search_knn_vector_xd(query=feature, knn=k)
            for j in idx:
                match_inds.append([i, j])

        return np.asarray(match_inds)

    def get_points_inds(
            self,
            source,
            target,
            k=30
    ):
        source_copy = o3d.geometry.PointCloud()
        source_copy.points = o3d.utility.Vector3dVector(source)
        target_copy = o3d.geometry.PointCloud()
        target_copy.points = o3d.utility.Vector3dVector(target)

        kdtree = o3d.geometry.KDTreeFlann(target_copy)
        match_inds = []
        for i, point in enumerate(source_copy.points):
            [_, idx, _] = kdtree.search_knn_vector_3d(query=point,knn=k)
            for j in idx:
                match_inds.append([i, j])

        return np.asarray(match_inds)

    def get_best_feature(self,p_feature,p_xyz,q_feature,q_xyz,k=10):

        feature_inds = self.get_feature_inds(source=p_feature,target=q_feature,k=k)
        points_inds = self.get_points_inds(source=p_xyz,target=q_xyz,k=k)
        best_feature_index = -1
        best_feature_match = -1
        best_feature_p_ind = []
        best_feature_q_ind = []
        for i in range(len(p_xyz)):
            p_inds = feature_inds[feature_inds[:,0] == i,:]
            q_inds = points_inds[points_inds[:,0] == i,:]
            match = np.sum(p_inds==q_inds)
            if(match > best_feature_match):
                best_feature_index = i
                best_feature_match = match
                best_feature_p_ind = p_inds
                best_feature_q_ind = q_inds

        print(f"Best Feature Index:{best_feature_index},Matching:{best_feature_match} points,K:{k}")

        return [best_feature_index]

    def get_dam_image(
            self,
            input_tensor,
            target_category,
            activations,
            grads,
    ):
        # the feature map channel importance
        weights = self.get_cam_weights(input_tensor, target_category, activations, grads)
        weighted_activations = weights[:,None] * activations
        dam = weighted_activations.sum(axis=0)
        # importance
        return dam

    # get the map
    def forward(
            self,
            input_tensor,
            target_point_index=None,
            target_layer=None
    ):

        # feature map
        p_feature,p_coord = self.activations_and_grads(input_tensor)

        # 如果目标类别是None，则使用标签中类别分数最大的；否则用指定的类别
        if type(target_point_index) is int:
            target_point_index = [target_point_index]
        # elif target_point_index is None:
        #     target_point_index = self.get_best_feature(
        #         p_feature=p_feature.cpu().data.numpy(),
        #         p_xyz = p_xyz,
        #         q_feature=q_feature.cpu().data.numpy(),
        #         q_xyz=q_xyz,
        #         k=10
        #     )

        self.model.zero_grad()
        # get the loss from the desc channel elements
        losses,coord = self.get_loss(
            p_feature = p_feature,
            p_xyz = p_coord,
            target_category = target_point_index
        )

        dams = []
        for i in range(losses.shape[1]):
            loss = losses[0,i]
            loss.backward(retain_graph=True)
            kernal_grad = target_layer.kernel.grad.cpu().detach().numpy()
            output_activation = self.activations_and_grads.output_activations[-1].cpu().data.numpy()
            output_activation = np.transpose(output_activation,axes=(1,0))
            feature_grad = np.transpose(kernal_grad,axes=(1,0))

            dam = self.get_dam_image(
                input_tensor,
                target_point_index,
                output_activation,
                feature_grad,
            )

            dams.append(dam)
        # add all important weight
        dams = np.asarray(dams)
        dam_weight = np.sum(dams,axis=0)
        # ReLU
        dam_weight = np.maximum(dam_weight, 0)
        # the weight of pc
        return dam_weight, target_point_index[0]


    def __call__(self,
                 input_tensor,
                 target_category=None,
                 target_layer=None):

        return self.forward(
            input_tensor,
            target_category,
            target_layer
        )