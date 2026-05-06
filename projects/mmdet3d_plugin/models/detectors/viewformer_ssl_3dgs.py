'''
Copyright (c) 2024 by Haiming Zhang. All Rights Reserved.

Author: Haiming Zhang
Date: 2024-07-11 10:42:14
Email: haimingzhang@link.cuhk.edu.cn
Description: Use the 3DGS as the pretraining decoder.
'''
import torch
import numpy as np
import torch.nn.functional as F

from mmcv.runner import force_fp32, auto_fp16
from mmdet.models import DETECTORS
from mmdet3d.models.detectors.mvx_two_stage import MVXTwoStageDetector
from projects.mmdet3d_plugin.models.utils.grid_mask import GridMask

from .viewformer_ssl import ViewFormerSSL

@DETECTORS.register_module()
class ViewFormerSSL3DGS(ViewFormerSSL):
    """UVTRSSL3DGS."""

    def __init__(
        self,
        **kwargs,
    ):
        super(ViewFormerSSL3DGS, self).__init__(**kwargs)

    # def forward_train(self, 
    #                   points=None, 
    #                   img_metas=None, 
    #                   img=None,
    #                   **kwargs):
    #     """Forward training function.
    #     Returns:
    #         dict: Losses of different branches.
    #     """
    #     pts_feats, img_feats, img_depth = self.extract_feat(
    #         points=points, img=img, img_metas=img_metas
    #     )
    #     losses = dict()
    #     losses_pts = self.forward_pts_train(
    #         pts_feats, img_feats, points, img, img_metas, img_depth,
    #         **kwargs
    #     )
    #     losses.update(losses_pts)
    #     return losses
    
    @force_fp32(apply_to=("pts_feats"))
    def forward_pts_train(self,
                          pts_feats,
                          gt_bboxes_3d=None,
                          gt_labels_3d=None,
                          gt_bboxes_ignore=None,
                          img_metas=None,
                          prev_exists=None,
                          voxel_semantics=None,
                          mask_lidar=None,
                          mask_camera=None,
                          requires_grad=True,
                          return_losses=False,
                          **kwargs
                          ):
        """Forward function for point cloud branch.
        Args:
            pts_feats (list[torch.Tensor]): Features of point cloud branch
            gt_bboxes_3d (list[:obj:`BaseInstance3DBoxes`]): Ground truth
                boxes for each sample.
            gt_labels_3d (list[torch.Tensor]): Ground truth labels for
                boxes of each sampole
            img_metas (list[dict]): Meta information of samples.
            gt_bboxes_ignore (list[torch.Tensor], optional): Ground truth
                boxes to be ignored. Defaults to None.
        Returns:
            dict: Losses of each branch.
        """

        if not requires_grad:
            self.eval()
            with torch.no_grad():
                outs = self.pts_bbox_head(pts_feats,
                                          img_metas,
                                          prev_exists=prev_exists,
                                          bev_only=not return_losses)
            self.train()
        else:
            outs = self.pts_bbox_head(pts_feats,
                                      img_metas,
                                      prev_exists=prev_exists,
                                      bev_only=not return_losses)

        # if return_losses:
        #     loss_inputs = [voxel_semantics, mask_lidar, mask_camera, outs]
        #     losses = self.pts_bbox_head.loss(*loss_inputs, img_metas=img_metas)

        #     return losses, outs['bev_embed']
        # else:
        #     return None, outs['bev_embed']

        target_dict = dict(**kwargs)
        losses = self.pts_bbox_head.loss(outs, target_dict)

        return losses, None

    # def simple_test(self, img_metas, points=None, img=None, **kwargs):
    #     """Test function without augmentaiton."""
    #     pts_feat, img_feats, img_depth = self.extract_feat(
    #         points=points, img=img, img_metas=img_metas
    #     )
    #     self.pts_bbox_head.vis_pred = True
    #     results = self.pts_bbox_head(
    #         pts_feat, img_feats, img_metas, 
    #         img_depth, img, **kwargs
    #     )
    #     # set_trace()
    #     return [results]