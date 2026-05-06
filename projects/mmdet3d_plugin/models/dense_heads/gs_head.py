'''
Copyright (c) 2024 by Haiming Zhang. All Rights Reserved.

Author: Haiming Zhang
Date: 2024-07-10 10:45:56
Email: haimingzhang@link.cuhk.edu.cn
Description: The 3D Gaussian Splatting rendering head.
'''

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import cv2
import torch_scatter
import os.path as osp
from einops import rearrange, repeat
from mmdet3d.models.builder import HEADS
from mmcv.runner.base_module import BaseModule
# from mmdet3d.models.decode_heads.nerf_head import NeRFDecoderHead
from .common.gaussians import build_covariance
from .common.cuda_splatting import render_cuda, render_depth_cuda, render_depth_cuda2
from .common.sh_rotation import rotate_sh
# from .gs_utils import get_rays_of_a_view


class Gaussians:
    means: torch.FloatTensor
    covariances: torch.FloatTensor
    scales: torch.FloatTensor
    rotations: torch.FloatTensor
    harmonics: torch.FloatTensor
    opacities: torch.FloatTensor
    feats: torch.FloatTensor
    means_shift: torch.FloatTensor


@HEADS.register_module()
class GaussianSplattingDecoder(BaseModule):
    def __init__(self,
                 semantic_head=False,
                 render_size=None,
                 depth_range=None,
                 depth_loss_type='l1',
                 pc_range=None,
                 learn_gs_scale_rot=False,
                 gs_scale_min=0.1,
                 gs_scale_max=0.24,
                 sh_degree=4,
                 volume_size=(200, 200, 16),
                 in_channels=32,
                 num_surfaces=1,
                 num_offsets=1,
                 offset_scale=0.05,
                 gs_scale=0.05,
                 rescale_z_axis=False,
                 vis_gt=False,
                 use_depth_loss=True,
                 filter_opacities=False,
                 use_old_predictor=False,
                 learn_gs_offset=True,
                 pred_density=False,

                 use_reliev3R=False,
                 use_sc_depth=False,

                 **kwargs):
        super().__init__()

        self.render_h, self.render_w = render_size
        self.min_depth, self.max_depth = depth_range

        self.use_depth_loss = use_depth_loss
        self.filter_opacities = filter_opacities
        self.learn_gs_offset = learn_gs_offset
        self.num_offsets = num_offsets

        self.in_channels = in_channels

        self.gs_mask = 'depth'

        self.depth_loss_type = depth_loss_type

        self.loss_weight = [1.0, 1.0, 1.0]

        self.semantic_head = semantic_head
        self.img_recon_head = False
        self.vis_gt = vis_gt  # NOTE: FOR DEBUG

        self.learn_gs_scale_rot = learn_gs_scale_rot
        self.offset_scale = offset_scale
        self.gs_scale = gs_scale
        self.rescale_z_axis = rescale_z_axis

        # NOTE: due to history reason, we predict the density before this module, here
        # we can choose to predict the density in this module
        self.pred_density = pred_density

        self.xyz_min = torch.from_numpy(np.array(pc_range[:3]))  # (x_min, y_min, z_min)
        self.xyz_max = torch.from_numpy(np.array(pc_range[3:]))  # (x_max, y_max, z_max)

        ## construct the volume grid
        xs = torch.arange(
            self.xyz_min[0], self.xyz_max[0],
            (self.xyz_max[0] - self.xyz_min[0]) / volume_size[0])
        ys = torch.arange(
            self.xyz_min[1], self.xyz_max[1],
            (self.xyz_max[1] - self.xyz_min[1]) / volume_size[1])
        zs = torch.arange(
            self.xyz_min[2], self.xyz_max[2],
            (self.xyz_max[2] - self.xyz_min[2]) / volume_size[2])
        W, H, D = len(xs), len(ys), len(zs)

        xyzs = torch.stack([
            xs[None, :, None].expand(H, W, D),
            ys[:, None, None].expand(H, W, D),
            zs[None, None, :].expand(H, W, D)
        ], dim=-1).permute(1, 0, 2, 3)  # (200, 200, 16, 3)

        # the volume grid coordinates in ego frame
        self.volume_xyz = xyzs.to(torch.float32)

        self.gs_scale_min = gs_scale_min
        self.gs_scale_max = gs_scale_max
        self.d_sh = (sh_degree + 1) ** 2

        self.register_buffer(
            "sh_mask",
            torch.ones((self.d_sh,), dtype=torch.float32),
            persistent=False,
        )
        for degree in range(1, sh_degree + 1):
            self.sh_mask[degree**2 : (degree + 1) ** 2] = 0.1 * 0.25**degree

        self.OCC3D_PALETTE = torch.Tensor([
            [0, 0, 0],
            [255, 120, 50],  # barrier              orangey
            [255, 192, 203],  # bicycle              pink
            [255, 255, 0],  # bus                  yellow
            [0, 150, 245],  # car                  blue
            [0, 255, 255],  # construction_vehicle cyan
            [200, 180, 0],  # motorcycle           dark orange
            [255, 0, 0],  # pedestrian           red
            [255, 240, 150],  # traffic_cone         light yellow
            [135, 60, 0],  # trailer              brown
            [160, 32, 240],  # truck                purple
            [255, 0, 255],  # driveable_surface    dark pink
            [139, 137, 137], # other_flat           dark grey
            [75, 0, 75],  # sidewalk             dard purple
            [150, 240, 80],  # terrain              light green
            [230, 230, 250],  # manmade              white
            [0, 175, 0],  # vegetation           green
            [0, 255, 127],  # ego car              dark cyan
            [255, 99, 71],
            [0, 191, 255],
            [125, 125, 125]
        ])

        extra_dim = 1 if self.pred_density else 0

        if use_old_predictor:
            # NOTE: just for back compatibility
            self.to_gaussians = nn.Sequential(
                nn.ReLU(),
                nn.Linear(
                    in_channels,
                    num_surfaces * (extra_dim + 3 + 3 + 4 + 3 * self.d_sh)
                )
            )
        else:
            # we need predict the opacities for each offset here
            self.to_gaussians = nn.Sequential(
                nn.Linear(in_channels, 64),
                nn.ReLU(),
                nn.Linear(
                    64,
                    num_surfaces * num_offsets * (
                        extra_dim + 3 + 3 + 4 + 3 * self.d_sh)
                )
            )

        self.use_reliev3R = use_reliev3R
        self.use_sc_depth = use_sc_depth

    def forward(self, 
                inputs,
                vis_gt=False,
                return_gaussians=False,
                suffix='',
                **kwargs):
        """Foward function

        Args:
            inputs: (dict), including density_prob (Tensor): (bs, 1, 200, 200, 16)
            rgb_recon (Tensor): (bs, 3, 200, 200, 16)
            occ_semantic (Tensor): (bs, c, 200, 200, 16)
            intricics (Tensor): (bs, num_view, 4, 4)
            pose_spatial (Tensor): (bs, num_view, 4, 4)
            volume_feat (Tensor): (bs, 200, 200, 16, c)
            render_mask (_type_, optional): _description_. Defaults to None.

        Returns:
            Tuple: rendered depth, rgb images and semantic features
        """
        # get occupancy features
        density_prob = inputs['density_prob']  # B, 1, X, Y, Z
        semantic = inputs['semantic'] # B, c, X, Y, Z

        intricics, pose_spatial = inputs['intrinsics'], inputs['pose_spatial']

        if vis_gt:
            semantic_dummy = repeat(semantic[:, 0:1], 'b dim1 x y z -> b (dim1 C) x y z', C=17).float()
            semantic_dummy = torch.rand(semantic_dummy.shape).to(semantic_dummy.device)
            with torch.no_grad():
                render_depth, render_rgb, render_semantic = self.visualize_gaussian(
                    density_prob,
                    semantic,
                    semantic_dummy,
                    intricics,
                    pose_spatial
                )
            return render_depth, render_rgb, render_semantic
        
        volume_feat = inputs['volume_feat']  # B, X, Y, Z, C

        # offsets = None
        # if 'offsets' in inputs:
        #     offsets = inputs['offsets']
        #     if semantic is not None:
        #         semantic = semantic.repeat(intricics.shape[0], 1, 1, 1, 1)

        if self.use_reliev3R or self.use_sc_depth:
            if semantic is not None:
                semantic = semantic.repeat(intricics.shape[0], 1, 1, 1, 1)


        render_depth, render_rgb, render_semantic, gaussians = \
            self.train_gaussian_rasterization_v2(
                density_prob,
                None,
                semantic,
                intricics,
                pose_spatial,
                volume_feat=volume_feat,
                # offsets=offsets,
                inputs=inputs,
        )
        render_depth = render_depth.clamp(self.min_depth, self.max_depth)

        dec_output = {'render_depth' + suffix: render_depth,
                      'render_rgb' + suffix: render_rgb,
                      'render_semantic' + suffix: render_semantic}
        if return_gaussians:
            return dec_output, gaussians
        
        return dec_output
    
    def render_results(self,
                       inputs,
                       gaussians):
        """Render the Gaussian splatting results

        Args:
            inputs (dict): input dict
            gaussians (Gaussians): Gaussian parameters
            suffix (str, optional): suffix for the output. Defaults to ''.

        Returns:
            dict: rendered results
        """
        intrinsics = inputs['intrinsics']
        extrinsics = inputs['pose_spatial']

        b, v = intrinsics.shape[:2]
        device = gaussians.means.device
        
        near = torch.ones(b, v).to(device) * self.min_depth
        far = torch.ones(b, v).to(device) * self.max_depth
        background_color = torch.zeros((3), dtype=torch.float32).to(device)
        
        intrinsics = intrinsics[..., :3, :3]
        # normalize the intrinsics
        intrinsics[:, :, 0] /= self.render_w
        intrinsics[:, :, 1] /= self.render_h

        # use the new extrinsics to compute the Gaussian covariances
        covariances = build_covariance(gaussians.scales, gaussians.rotations)
        covariances = rearrange(covariances, "b g i j -> b () g i j")

        c2w_rotations = extrinsics[..., :3, :3]
        c2w_rotations = rearrange(c2w_rotations, "b v i j -> b v () i j")
        covariances = c2w_rotations @ covariances @ c2w_rotations.transpose(-1, -2)
        gaussians.covariances = covariances  # (bs, v, g, i, j)

        # start rendering
        render_results = render_cuda(
            rearrange(extrinsics, "b v i j -> (b v) i j"),
            rearrange(intrinsics, "b v i j -> (b v) i j"),
            rearrange(near, "b v -> (b v)"),
            rearrange(far, "b v -> (b v)"),
            (self.render_h, self.render_w),
            repeat(background_color, "c -> (b v) c", b=b, v=v),
            repeat(gaussians.means, "b g xyz -> (b v) g xyz", v=v),
            rearrange(gaussians.covariances, "b v g i j -> (b v) g i j"),
            rearrange(gaussians.harmonics, "b v g c d_sh -> (b v) g c d_sh"),
            repeat(gaussians.opacities, "b g -> (b v) g", v=v),
            scale_invariant=False,
            use_sh=True,
            feats3D=gaussians.feats
        )
        return render_results
    
    def render_forward(self, 
                       inputs,
                       gaussians,
                       suffix='',):
        
        intrinsics = inputs['intrinsics']
        extrinsics = inputs['pose_spatial']

        b, v = intrinsics.shape[:2]
        device = gaussians.means.device
        
        near = torch.ones(b, v).to(device) * self.min_depth
        far = torch.ones(b, v).to(device) * self.max_depth
        background_color = torch.zeros((3), dtype=torch.float32).to(device)
        
        intrinsics = intrinsics[..., :3, :3]
        # normalize the intrinsics
        intrinsics[:, :, 0] /= self.render_w
        intrinsics[:, :, 1] /= self.render_h

        # use the new extrinsics to compute the Gaussian covariances
        covariances = build_covariance(gaussians.scales, gaussians.rotations)
        covariances = rearrange(covariances, "b g i j -> b () g i j")

        c2w_rotations = extrinsics[..., :3, :3]
        c2w_rotations = rearrange(c2w_rotations, "b v i j -> b v () i j")
        covariances = c2w_rotations @ covariances @ c2w_rotations.transpose(-1, -2)
        gaussians.covariances = covariances  # (bs, v, g, i, j)

        # start rendering
        render_results = render_cuda(
            rearrange(extrinsics, "b v i j -> (b v) i j"),
            rearrange(intrinsics, "b v i j -> (b v) i j"),
            rearrange(near, "b v -> (b v)"),
            rearrange(far, "b v -> (b v)"),
            (self.render_h, self.render_w),
            repeat(background_color, "c -> (b v) c", b=b, v=v),
            repeat(gaussians.means, "b g xyz -> (b v) g xyz", v=v),
            rearrange(gaussians.covariances, "b v g i j -> (b v) g i j"),
            rearrange(gaussians.harmonics, "b v g c d_sh -> (b v) g c d_sh"),
            repeat(gaussians.opacities, "b g -> (b v) g", v=v),
            scale_invariant=False,
            use_sh=True,
            feats3D=gaussians.feats
        )
        
        color, depth, feats = render_results
        if self.semantic_head:
            feats = rearrange(feats, "(b v) c h w -> b v c h w", b=b, v=v)
        else:
            feats = None
        
        color = rearrange(color, "(b v) c h w -> b v c h w", b=b, v=v)
        depth = rearrange(depth, "(b v) c h w -> b v c h w", b=b, v=v).squeeze(2)
        
        depth = depth.clamp(self.min_depth, self.max_depth)

        dec_output = {'render_depth' + suffix: depth,
                      'render_rgb' + suffix: color,
                      'render_semantic' + suffix: feats}
        return dec_output

    def train_gaussian_rasterization(self, 
                                     density_prob, 
                                     rgb_recon, 
                                     semantic_pred, 
                                     intrinsics, 
                                     extrinsics, 
                                     render_mask=None,
                                     vis_semantic=False,
                                     **kwargs):
        b, v = intrinsics.shape[:2]
        device = density_prob.device
        
        near = torch.ones(b, v).to(device) * self.min_depth
        far = torch.ones(b, v).to(device) * self.max_depth
        background_color = torch.zeros((3), dtype=torch.float32).to(device)
        
        intrinsics = intrinsics[..., :3, :3]
        # normalize the intrinsics
        intrinsics[..., 0, :] /= self.render_w
        intrinsics[..., 1, :] /= self.render_h

        transform = torch.Tensor([[0, 1, 0, 0],
                                  [1, 0, 0, 0],
                                  [0, 0, 1, 0],
                                  [0, 0, 0, 1]]).to(device)
        extrinsics = transform.unsqueeze(0).unsqueeze(0) @ extrinsics
        
        bs = density_prob.shape[0]

        xyzs = repeat(self.volume_xyz, 'h w d dim3 -> bs h w d dim3', bs=bs).to(device)
        xyzs = rearrange(xyzs, 'b h w d dim3 -> b (h w d) dim3') # (bs, num, 3)

        if self.semantic_head:
            semantic_pred = rearrange(semantic_pred, 'b c h w d -> b (h w d) c').float()
            # semantic_pred = repeat(semantic_pred, 'b xyz c -> b xyz (dim17 c)', dim17=17)

        density_prob = rearrange(density_prob, 'b dim1 h w d -> (b dim1) (h w d)')
        
        if vis_semantic:
            harmonics = rearrange(rgb_recon, 'b dim3 h w d -> b (h w d) dim3 ()')
        else:
            ## TODO: currently the harmonics is a dummy variable when training
            harmonics = self.OCC3D_PALETTE[torch.argmax(rgb_recon, dim=1).long()].to(device)
            harmonics = rearrange(harmonics, 'b h w d dim3 -> b (h w d) dim3 ()')

        g = xyzs.shape[1]

        gaussians = Gaussians
        gaussians.means = xyzs  ######## Gaussian center ########
        gaussians.opacities = torch.sigmoid(density_prob) ######## Gaussian opacities ########

        scales = torch.ones(3).unsqueeze(0).to(device) * 0.2
        rotations = torch.Tensor([1, 0, 0, 0]).unsqueeze(0).to(device)

        # Create world-space covariance matrices.
        covariances = build_covariance(scales, rotations)
        c2w_rotations = extrinsics[..., :3, :3]
        covariances = c2w_rotations @ covariances @ c2w_rotations.transpose(-1, -2)
        gaussians.covariances = covariances ######## Gaussian covariances ########

        gaussians.harmonics = harmonics ######## Gaussian harmonics ########

        render_results = render_cuda(
            rearrange(extrinsics, "b v i j -> (b v) i j"),
            rearrange(intrinsics, "b v i j -> (b v) i j"),
            rearrange(near, "b v -> (b v)"),
            rearrange(far, "b v -> (b v)"),
            (self.render_h, self.render_w),
            repeat(background_color, "c -> (b v) c", b=b, v=v),
            repeat(gaussians.means, "b g xyz -> (b v) g xyz", v=v),
            repeat(gaussians.covariances, "b v i j -> (b v) g i j", g=g),
            repeat(gaussians.harmonics, "b g c d_sh -> (b v) g c d_sh", v=v),
            repeat(gaussians.opacities, "b g -> (b v) g", v=v),
            scale_invariant=False,
            use_sh=False,
            feats3D=repeat(semantic_pred, "b g c -> (b v) g c", v=v) if self.semantic_head else None,
        )
        if self.semantic_head:
            color, depth, feats = render_results
            feats = rearrange(feats, "(b v) c h w -> b v c h w", b=b, v=v)
        else:
            color, depth = render_results
            feats = None
        
        color = rearrange(color, "(b v) c h w -> b v c h w", b=b, v=v)
        depth = rearrange(depth, "(b v) c h w -> b v c h w", b=b, v=v).squeeze(2)

        return depth, color, feats
    
    def loss(self,
             pred_dict,
             target_dict,
             weight=1.0,
             suffix=''):
        
        losses = dict()

        if self.use_depth_loss:
            render_depth = pred_dict['render_depth' + suffix]
            gt_depth = target_dict['render_gt_depth' + suffix]

            loss_render_depth = self.compute_depth_loss(
                render_depth, gt_depth, gt_depth > 0.0)
            if torch.isnan(loss_render_depth):
                print('NaN in render depth loss!')
                loss_render_depth = torch.Tensor([0.0]).to(render_depth.device)
            losses['loss_render_depth' + suffix] = weight * loss_render_depth

        if self.semantic_head:
            assert 'render_gt_semantic' in target_dict.keys()
            semantic_gt = target_dict['render_gt_semantic' + suffix]

            semantic_pred = pred_dict['render_semantic' + suffix]
            
            loss_render_sem = self.compute_semantic_loss(
                semantic_pred, semantic_gt, ignore_index=255)
            if torch.isnan(loss_render_sem):
                print('NaN in render semantic loss!')
                loss_render_sem = torch.Tensor([0.0]).to(render_depth.device)
            losses['loss_render_sem' + suffix] = weight * loss_render_sem

        ## Compute the sigma loss and sdf loss, TODO
        return losses

    def train_gaussian_rasterization_v2(self, 
                                        density_prob, 
                                        rgb_recon, 
                                        semantic_pred, 
                                        intrinsics, 
                                        extrinsics, 
                                        volume_feat,
                                        # offsets=None
                                        inputs=None,
                                        ):
        b, v = intrinsics.shape[:2]
        device = volume_feat.device
        
        near = torch.ones(b, v).to(device) * self.min_depth
        far = torch.ones(b, v).to(device) * self.max_depth
        background_color = torch.zeros((3), dtype=torch.float32).to(device)
        
        intrinsics = intrinsics[..., :3, :3]
        # normalize the intrinsics
        intrinsics[:, :, 0] /= self.render_w
        intrinsics[:, :, 1] /= self.render_h

        # if self.semantic_head and semantic_pred is not None:
        if semantic_pred is not None:
            semantic_pred = rearrange(semantic_pred, 'b c h w d -> b (h w d) c')
            _feats3D = repeat(semantic_pred, "b g c -> (b v) g c", v=v)
        else:
            _feats3D = None

        if self.num_offsets == 1:
            if density_prob is not None:
                density_prob = rearrange(density_prob, 'b dim1 h w d -> b (h w d dim1)')
            gaussians = self.predict_gaussian(density_prob,
                                              extrinsics,
                                              volume_feat,
                                            #   offsets
                                              inputs
                                              )
        else:
            if density_prob is not None: 
                density_prob = rearrange(density_prob, 'b num h w d -> b (h w d num)') # (b, g)
            gaussians = self.predict_gaussian_v2(density_prob,
                                                 extrinsics,
                                                 volume_feat)
        
        gaussians.feats = _feats3D

        if self.filter_opacities:
            mask = (gaussians.opacities > 0.0)
            # set the opacities to 0.0 if the opacities are less than 0.0
            gaussians.opacities = gaussians.opacities * mask
        
        # start rendering
        render_results = render_cuda(
            rearrange(extrinsics, "b v i j -> (b v) i j"),
            rearrange(intrinsics, "b v i j -> (b v) i j"),
            rearrange(near, "b v -> (b v)"),
            rearrange(far, "b v -> (b v)"),
            (self.render_h, self.render_w),
            repeat(background_color, "c -> (b v) c", b=b, v=v),
            repeat(gaussians.means, "b g xyz -> (b v) g xyz", v=v),
            rearrange(gaussians.covariances, "b v g i j -> (b v) g i j"),
            rearrange(gaussians.harmonics, "b v g c d_sh -> (b v) g c d_sh"),
            repeat(gaussians.opacities, "b g -> (b v) g", v=v),
            scale_invariant=False,
            use_sh=True,
            feats3D=gaussians.feats
        )
        
        color, depth, feats = render_results
        # if self.semantic_head and semantic_pred is not None:
        if semantic_pred is not None:
            feats = rearrange(feats, "(b v) c h w -> b v c h w", b=b, v=v)
        else:
            feats = None
        
        color = rearrange(color, "(b v) c h w -> b v c h w", b=b, v=v)
        depth = rearrange(depth, "(b v) c h w -> b v c h w", b=b, v=v).squeeze(2)

        return depth, color, feats, gaussians
    

    def visualize_gaussian(self,
                           density_prob, 
                           rgb_recon, 
                           semantic_pred, 
                           intrinsics, 
                           extrinsics):
        b, v = intrinsics.shape[:2]
        device = density_prob.device
        
        near = torch.ones(b, v).to(device) * self.min_depth
        far = torch.ones(b, v).to(device) * self.max_depth
        background_color = torch.zeros((3), dtype=torch.float32).to(device)
        
        intrinsics = intrinsics[..., :3, :3]
        # normalize the intrinsics
        intrinsics[..., 0, :] /= self.render_w
        intrinsics[..., 1, :] /= self.render_h

        bs = density_prob.shape[0]
        xyzs = repeat(self.volume_xyz, 'h w d dim3 -> bs h w d dim3', bs=bs).to(device)
        xyzs = rearrange(xyzs, 'b h w d dim3 -> b (h w d) dim3') # (bs, num, 3)

        density_prob = rearrange(density_prob, 'b dim1 h w d -> (b dim1) (h w d)')

        if self.semantic_head:
            semantic_pred = rearrange(semantic_pred, 'b c h w d -> b (h w d) c')

        harmonics = rearrange(rgb_recon, 'b dim3 h w d -> b (h w d) dim3 ()')
        g = xyzs.shape[1]

        gaussians = Gaussians
        gaussians.means = xyzs  ######## Gaussian center ########
        gaussians.opacities = torch.sigmoid(density_prob) ######## Gaussian opacities ########

        scales = torch.ones(3).unsqueeze(0).to(device) * 0.2
        rotations = torch.Tensor([1, 0, 0, 0]).unsqueeze(0).to(device)

        # Create world-space covariance matrices.
        covariances = build_covariance(scales, rotations)
        c2w_rotations = extrinsics[..., :3, :3]
        covariances = c2w_rotations @ covariances @ c2w_rotations.transpose(-1, -2)
        gaussians.covariances = covariances ######## Gaussian covariances ########

        gaussians.harmonics = harmonics ######## Gaussian harmonics ########

        render_results = render_cuda(
            rearrange(extrinsics, "b v i j -> (b v) i j"),
            rearrange(intrinsics, "b v i j -> (b v) i j"),
            rearrange(near, "b v -> (b v)"),
            rearrange(far, "b v -> (b v)"),
            (self.render_h, self.render_w),
            repeat(background_color, "c -> (b v) c", b=b, v=v),
            repeat(gaussians.means, "b g xyz -> (b v) g xyz", v=v),
            repeat(gaussians.covariances, "b v i j -> (b v) g i j", g=g),
            repeat(gaussians.harmonics, "b g c d_sh -> (b v) g c d_sh", v=v),
            repeat(gaussians.opacities, "b g -> (b v) g", v=v),
            scale_invariant=False,
            use_sh=False,
            feats3D=repeat(semantic_pred, "b g c -> (b v) g c", v=v)
        )
        if self.semantic_head:
            color, depth, feats = render_results
            feats = rearrange(feats, "(b v) c h w -> b v c h w", b=b, v=v)
        else:
            color, depth = render_results
            feats = None
        
        color = rearrange(color, "(b v) c h w -> b v c h w", b=b, v=v)
        depth = rearrange(depth, "(b v) c h w -> b v c h w", b=b, v=v).squeeze(2)

        return depth, color, feats

    def predict_gaussian(self,
                         density_prob,
                         extrinsics,
                         volume_feat,
                        #  offsets=None
                         inputs=None,
                         ):
        """Learn the 3D Gaussian parameters from the volume feature

        Args:
            density_prob (Tesnro): (bs, g)
            extrinsics (Tensor): (bs, v, 4, 4)
            volume_feat (Tensor): (bs, h, w, d, c)

        Returns:
            class: Gaussians class containing the Gaussian parameters
        """
        bs, v = extrinsics.shape[:2]
        device = extrinsics.device
        
        xyzs = repeat(self.volume_xyz, 'h w d dim3 -> bs h w d dim3', bs=bs).to(device)
        xyzs = rearrange(xyzs, 'b h w d dim3 -> b (h w d) dim3') # (bs, num, 3)

        # predict the Gaussian parameters from volume feature
        raw_gaussians = self.to_gaussians(volume_feat)
        # if offsets is not None:
        #     raw_gaussians = raw_gaussians.repeat(bs, 1, 1, 1, 1)
        #     density_prob = density_prob.repeat(bs, 1)

        if self.use_reliev3R or self.use_sc_depth:
            raw_gaussians = raw_gaussians.repeat(bs, 1, 1, 1, 1)
            density_prob = density_prob.repeat(bs, 1)

        raw_gaussians = rearrange(raw_gaussians, 'b h w d c -> b (h w d) c')
        if not self.pred_density:
            xyz_offset, scales, rotations, sh = raw_gaussians.split(
                (3, 3, 4, 3 * self.d_sh), dim=-1)
        else:
            density_prob, xyz_offset, scales, rotations, sh = raw_gaussians.split(
                (1, 3, 3, 4, 3 * self.d_sh), dim=-1)
        
        density_prob = density_prob.squeeze(-1)

        # construct 3D Gaussians
        gaussians = Gaussians
        
        if not self.filter_opacities:
            opacities = torch.sigmoid(density_prob)
        else:
            opacities = torch.tanh(density_prob)  # to (-1, 1)
        gaussians.opacities = opacities

        if self.use_reliev3R or self.use_sc_depth:
            means = xyzs + (xyz_offset.squeeze(0).sigmoid() - 0.5) * self.offset_scale  # [3, 162000, 3]

            # if offsets is not None:
            if 'offsets' in inputs:
                offsets = inputs['offsets']
                # means = xyzs + (xyz_offset.squeeze(0).sigmoid() - 0.5) * self.offset_scale
                offsets = offsets.permute(0, 2, 1, 3).contiguous()
                offsets = offsets.view(-1, *offsets.shape[2:])
                means[1:] = means[1:] + offsets     

            lidar2global = inputs['lidar2global'][0]     # [6, 3, 4, 4] -> [3, 4, 4]
            prev2cur = torch.matmul(torch.linalg.inv(lidar2global[1:]), lidar2global[0:1])[:, None, :, :] # 2, 1, 4, 4
            # if prev2cur.shape[1] < xyz.shape[1]:
            #     num_pad_frs = xyz.shape[1] - prev2cur.shape[1]
            #     prev2cur = torch.cat([prev2cur, torch.eye(4, dtype=prev2cur.dtype, device=prev2cur.device)[None, None, None].repeat(B, num_pad_frs, 1, 1, 1)], dim=1)
            new_xyz = torch.matmul(prev2cur, torch.cat([means[1:], torch.ones_like(means[1:][..., :1])], dim=-1)[..., None])[..., :3, 0]     # bs, f, g, 3 
            means = torch.cat([means[0:1], new_xyz])

            gaussians.means = means
        else:
            if self.learn_gs_offset:
                gaussians.means = xyzs + (xyz_offset.sigmoid() - 0.5) * self.offset_scale
            else:
                gaussians.means = xyzs

        # Learn scale and rotation of 3D Gaussians
        if self.learn_gs_scale_rot:
            # Set scale and rotation of 3D Gaussians
            scale_min = self.gs_scale_min
            scale_max = self.gs_scale_max
            scales = scale_min + (scale_max - scale_min) * torch.sigmoid(scales)

            # Normalize the quaternion features to yield a valid quaternion.
            rotations = rotations / (rotations.norm(dim=-1, keepdim=True) + 1e-8)
        else:
            scales = torch.ones(3).unsqueeze(0).unsqueeze(0).to(device) * self.gs_scale
            rotations = torch.Tensor([1, 0, 0, 0]).unsqueeze(0).unsqueeze(0).to(device)
            scales = scales.repeat(bs, xyzs.shape[1], 1)
            rotations = rotations.repeat(bs, xyzs.shape[1], 1)

        if self.rescale_z_axis:
            scales = scales * torch.tensor([1, 1, 2]).to(device)
        
        gaussians.scales = scales
        gaussians.rotations = rotations
        
        # Create world-space covariance matrices.
        covariances = build_covariance(scales, rotations)
        covariances = rearrange(covariances, "b g i j -> b () g i j")

        c2w_rotations = extrinsics[..., :3, :3]
        c2w_rotations = rearrange(c2w_rotations, "b v i j -> b v () i j")
        covariances = c2w_rotations @ covariances @ c2w_rotations.transpose(-1, -2)
        gaussians.covariances = covariances  # (bs, v, g, i, j)

        # Apply sigmoid to get valid colors.
        sh = rearrange(sh, "... (xyz d_sh) -> ... xyz d_sh", xyz=3)
        sh = sh.broadcast_to((*gaussians.opacities.shape, 3, self.d_sh)) * self.sh_mask
        gaussians.harmonics = repeat(sh, 'b g xyz d_sh -> b v g xyz d_sh', v=v)

        return gaussians

    def predict_gaussian_v2(self,
                            density_prob,
                            extrinsics,
                            volume_feat):
        """Learn the 3D Gaussian parameters from the volume feature, we assume
        we predict multiple Gaussians for each voxel.

        Args:
            density_prob (Tesnro): (bs, g, 1)
            extrinsics (Tensor): (bs, v, 4, 4)
            volume_feat (Tensor): (bs, h, w, d, c)

        Returns:
            class: Gaussians class containing the Gaussian parameters
        """
        bs, v = extrinsics.shape[:2]
        device = extrinsics.device

        xyzs = repeat(self.volume_xyz, 'h w d dim3 -> bs h w d dim3', bs=bs).to(device)
        xyzs = rearrange(xyzs, 'b h w d dim3 -> b (h w d) dim3') # (bs, num, 3)
        xyzs = repeat(xyzs, 'b g dim3 -> b (g num) dim3', num=self.num_offsets) # (bs, num, 3)
        
        # predict the Gaussian parameters from volume feature
        raw_gaussians = self.to_gaussians(volume_feat)
        raw_gaussians = rearrange(raw_gaussians, 'b h w d c -> b (h w d) c')
        xyz_offset, scales, rotations, sh = raw_gaussians.split(
            (3 * self.num_offsets, 3 * self.num_offsets, 
             4 * self.num_offsets, 3 * self.d_sh * self.num_offsets), dim=-1)
        
        xyz_offset = rearrange(xyz_offset, 'b g (num c) -> b (g num) c', num=self.num_offsets)
        scales = rearrange(scales, 'b g (num c) -> b (g num) c', num=self.num_offsets)
        rotations = rearrange(rotations, 'b g (num c) -> b (g num) c', num=self.num_offsets)
        sh = rearrange(sh, 'b g (num c) -> b (g num) c', num=self.num_offsets)

        # construct 3D Gaussians
        gaussians = Gaussians
        
        if not self.filter_opacities:
            opacities = torch.sigmoid(density_prob)
        else:
            opacities = torch.tanh(density_prob)  # to (-1, 1)
        gaussians.opacities = opacities

        if self.learn_gs_offset:
            gaussians.means = xyzs + (xyz_offset.sigmoid() - 0.5) * self.offset_scale
        else:
            gaussians.means = xyzs

        # Learn scale and rotation of 3D Gaussians
        if self.learn_gs_scale_rot:
            # Set scale and rotation of 3D Gaussians
            scale_min = self.gs_scale_min
            scale_max = self.gs_scale_max
            scales = scale_min + (scale_max - scale_min) * torch.sigmoid(scales)

            # Normalize the quaternion features to yield a valid quaternion.
            rotations = rotations / (rotations.norm(dim=-1, keepdim=True) + 1e-8)
        else:
            scales = torch.ones(3).unsqueeze(0).unsqueeze(0).to(device) * self.gs_scale
            rotations = torch.Tensor([1, 0, 0, 0]).unsqueeze(0).unsqueeze(0).to(device)
            scales = scales.repeat(bs, xyzs.shape[1], 1)
            rotations = rotations.repeat(bs, xyzs.shape[1], 1)

        if self.rescale_z_axis:
            scales = scales * torch.tensor([1, 1, 2]).to(device)
        
        gaussians.scales = scales
        gaussians.rotations = rotations
        
        # Create world-space covariance matrices.
        covariances = build_covariance(scales, rotations)
        covariances = rearrange(covariances, "b g i j -> b () g i j")

        c2w_rotations = extrinsics[..., :3, :3]
        c2w_rotations = rearrange(c2w_rotations, "b v i j -> b v () i j")
        covariances = c2w_rotations @ covariances @ c2w_rotations.transpose(-1, -2)
        gaussians.covariances = covariances  # (bs, v, g, i, j)

        # Apply sigmoid to get valid colors.
        sh = rearrange(sh, "... (xyz d_sh) -> ... xyz d_sh", xyz=3)
        sh = sh.broadcast_to((*gaussians.opacities.shape, 3, self.d_sh)) * self.sh_mask
        gaussians.harmonics = repeat(sh, 'b g xyz d_sh -> b v g xyz d_sh', v=v)

        return gaussians

    
    def gaussian_rasterization(self, 
                               density_prob, 
                               rgb_recon, 
                               semantic_pred, 
                               intrinsics, 
                               extrinsics, 
                               render_mask=None):
        b, v = intrinsics.shape[:2]
        
        near = torch.ones(b, v).to(density_prob.device) * 1
        far = torch.ones(b, v).to(density_prob.device) * 100
        background_color = torch.zeros((3), dtype=torch.float32).to(density_prob.device)
        
        intrinsics = intrinsics[..., :3, :3]
        # normalize the intrinsics
        intrinsics[..., 0, :] /= self.render_w
        intrinsics[..., 1, :] /= self.render_h

        transform = torch.Tensor([[0, 1, 0, 0],
                                  [1, 0, 0, 0],
                                  [0, 0, 1, 0],
                                  [0, 0, 0, 1]]).to(density_prob.device)
        extrinsics = transform.unsqueeze(0).unsqueeze(0) @ extrinsics

        device = density_prob.device
        xs = torch.arange(
            self.xyz_min[0], self.xyz_max[0],
            (self.xyz_max[0] - self.xyz_min[0]) / density_prob.shape[2], device=device)
        ys = torch.arange(
            self.xyz_min[1], self.xyz_max[1],
            (self.xyz_max[1] - self.xyz_min[1]) / density_prob.shape[3], device=device)
        zs = torch.arange(
            self.xyz_min[2], self.xyz_max[2],
            (self.xyz_max[2] - self.xyz_min[2]) / density_prob.shape[4], device=device)
        W, H, D = len(xs), len(ys), len(zs)
        
        bs = density_prob.shape[0]
        xyzs = torch.stack([
            xs[None, :, None].expand(H, W, D),
            ys[:, None, None].expand(H, W, D),
            zs[None, None, :].expand(H, W, D)
        ], dim=-1)[None].expand(bs, H, W, D, 3).flatten(0, 3)
        density_prob = density_prob.flatten()

        mask = (density_prob > 0) #& (semantic_pred.flatten()==3)
        xyzs = xyzs[mask]

        harmonics = self.OCC3D_PALETTE[semantic_pred.long().flatten()].to(device)
        harmonics = harmonics[mask]

        density_prob = density_prob[mask]
        density_prob = density_prob.unsqueeze(0)
        xyzs = xyzs.unsqueeze(0)

        g = xyzs.shape[1]

        gaussians = Gaussians
        gaussians.means = xyzs  ######## Gaussian center ########
        gaussians.opacities = torch.where(density_prob>0, 1., 0.) ######## Gaussian opacities ########

        scales = torch.ones(3).unsqueeze(0).to(device) * 0.05
        rotations = torch.Tensor([1, 0, 0, 0]).unsqueeze(0).to(device)

        # Create world-space covariance matrices.
        covariances = build_covariance(scales, rotations)
        c2w_rotations = extrinsics[..., :3, :3]
        covariances = c2w_rotations @ covariances @ c2w_rotations.transpose(-1, -2)
        gaussians.covariances = covariances ######## Gaussian covariances ########

        harmonics = harmonics.unsqueeze(-1).unsqueeze(0)
        # harmonics = torch.ones_like(xyzs).unsqueeze(-1)
        gaussians.harmonics = harmonics ######## Gaussian harmonics ########

        color, depth = render_cuda(
            rearrange(extrinsics, "b v i j -> (b v) i j"),
            rearrange(intrinsics, "b v i j -> (b v) i j"),
            rearrange(near, "b v -> (b v)"),
            rearrange(far, "b v -> (b v)"),
            (self.render_h, self.render_w),
            repeat(background_color, "c -> (b v) c", b=b, v=v),
            repeat(gaussians.means, "b g xyz -> (b v) g xyz", v=v),
            repeat(gaussians.covariances, "b v i j -> (b v) g i j", g=g),
            repeat(gaussians.harmonics, "b g c d_sh -> (b v) g c d_sh", v=v),
            repeat(gaussians.opacities, "b g -> (b v) g", v=v),
            scale_invariant=False,
            use_sh=False,
        )

        return color, depth.squeeze(1)
    
    def calculate_3dgs_loss(self, 
                            render_depth, depth_gt, depth_masks, 
                            semantic_pred, semantic_gt,
                            rgb_pred, render_img_gt):
        gs_loss, gs_depth, gs_sem, gs_img = [0.] * 4
        gs_depth = self.compute_depth_loss(render_depth, depth_gt, depth_masks)
        gs_mask = torch.ones_like(depth_masks).bool()
        if self.gs_mask == 'ego':
            msk_h = int(0.28 * depth_gt.shape[2])
            gs_mask[:, 0, -1 * msk_h:, :] = False
        elif self.gs_mask == 'depth':
            gs_mask = (depth_masks > 0)
            msk_h = int(0.28 * depth_gt.shape[2])
            gs_mask[:, 0, -1 * msk_h:, :] = False
        elif self.gs_mask == 'sky':
            gs_mask = (semantic_gt != 5)
            msk_h = int(0.28 * depth_gt.shape[2])
            gs_mask[:, 0, -1 * msk_h:, :] = False

        if torch.isnan(gs_depth):
            print('gs depth loss is nan!')
            gs_depth = torch.Tensor([0.]).cuda()
        gs_loss += gs_depth * self.loss_weight[0]

        if self.semantic_head:
            semantic_pred = semantic_pred[:, :semantic_gt.shape[1], :, :, :]
            if self.gs_mask is not None:
                gs_sem = self.compute_semantic_loss_flatten(
                    semantic_pred.permute(0, 1, 3, 4 ,2)[gs_mask], 
                    semantic_gt[gs_mask])
            else:
                gs_sem = self.compute_semantic_loss(semantic_pred, semantic_gt)

            if torch.isnan(gs_sem):
                print('gs semantic loss is nan!')
                gs_sem = torch.Tensor([0.]).cuda()
            gs_loss += gs_sem * self.loss_weight[1]

        if self.img_recon_head:
            if self.gs_mask is not None:
                gs_mask_img = gs_mask.unsqueeze(2).repeat(1, 1, 3, 1, 1)
                gs_img = self.compute_image_loss(
                    rgb_pred[gs_mask_img], render_img_gt[gs_mask_img])
            else:
                gs_img = self.compute_image_loss(rgb_pred, render_img_gt)

            if torch.isnan(gs_img):
                print('gs image loss is nan!')
                gs_img = torch.Tensor([0.]).cuda()
            gs_loss += gs_img * self.loss_weight[2]
            
        if self.overfit:
            print('gs_depth: {:4f}, gs_sem: {:4f}, gs_img: {:4f}'.format(
                gs_depth, gs_sem, gs_img))
        return gs_loss
    
    def compute_depth_loss(self, depth_est, depth_gt, mask):
        '''
        Args:
            mask: depth_gt > 0
        '''
        if self.depth_loss_type == 'silog':
            variance_focus = 0.85
            d = torch.log(depth_est[mask]) - torch.log(depth_gt[mask])
            loss = torch.sqrt((d ** 2).mean() - variance_focus * (d.mean() ** 2))
        elif self.depth_loss_type == 'l1':
            loss = F.l1_loss(depth_est[mask], depth_gt[mask])
        elif self.depth_loss_type == 'rl1':
            depth_est = (1 / depth_est) * self.max_depth
            depth_gt = (1 / depth_gt) * self.max_depth
            loss = F.l1_loss(depth_est[mask], depth_gt[mask], size_average=True)
        elif self.depth_loss_type == 'sml1':
            loss = F.smooth_l1_loss(depth_est[mask], depth_gt[mask], size_average=True)
        else:
            raise NotImplementedError()

        return loss

    def gaussian_sigma_sdf_loss(self, 
                                gaussians, 
                                intrinsics, 
                                pose_spatial, 
                                depths):
        batch_size, num_camera = intrinsics.shape[:2]
        intrinsics = intrinsics.view(-1, 4, 4)
        pose_spatial = pose_spatial.view(-1, 4, 4)

        with torch.no_grad():
            rays_o, rays_d = get_rays_of_a_view(
                H=self.render_h,
                W=self.render_w,
                K=intrinsics,
                c2w=pose_spatial,
                inverse_y=True,
                flip_x=False,
                flip_y=False,
                mode='center'
            )
        rays_o = rays_o.view(batch_size, num_camera, self.render_h, self.render_w, 3)
        rays_d = rays_d.view(batch_size, num_camera, self.render_h, self.render_w, 3)
        opacities = gaussians.opacities.view(batch_size, 1, *self.voxels_size)
        sigma_loss = 0
        sdf_estimation_loss = 0

        for b in range(batch_size):
            # sample pixels that depth > 0
            depth = depths[b]
            rays_o_i = rays_o[b][depth > 0]
            rays_d_i = rays_d[b][depth > 0]
            depth = depth[depth > 0]

            # sub-sample valid pixels
            rand_ind = torch.randperm(rays_o_i.shape[0])
            sampled_rays_o_i = rays_o_i[rand_ind][:self.max_ray_number]
            sampled_rays_d_i = rays_d_i[rand_ind][:self.max_ray_number]
            depth = depth[rand_ind][:self.max_ray_number]

            # calculate points on rays and interpolate gaussian opacities
            with torch.no_grad():
                rays_pts, mask_outbbox, interval, rays_pts_depth = \
                    self.sample_ray(sampled_rays_o_i, sampled_rays_d_i)
            sdf_estimation = interval - depth.unsqueeze(1)
            sdf_estimation = sdf_estimation[~mask_outbbox]
            mask_rays_pts = rays_pts[~mask_outbbox]
            interpolated_opacity = self.grid_sampler(mask_rays_pts, opacities[b])

            # calculate alpha values like NeRF
            interval_list = interval[..., 1:] - interval[..., :-1]
            alpha = torch.zeros_like(rays_pts[..., 0])
            alpha[~mask_outbbox] =  1 - torch.exp(-interpolated_opacity * interval_list[0, -1])
            alphainv_cum = torch.cat([torch.ones_like((1 - alpha)[..., [0]]), (1 - alpha).clamp_min(1e-10).cumprod(-1)], -1)
            weights = alpha * alphainv_cum[..., :-1]
            interval_list = torch.cat([interval_list, torch.Tensor([0]).to(weights.device).expand(interval_list[..., :1].shape)], -1)
            interval_list = interval_list * torch.norm(sampled_rays_d_i[..., None, :], dim=-1)

            # calculate sigma loss
            if self.sigma_loss_weight > 0:
                l = -torch.log(weights + 1e-5) * torch.exp(-(interval - depth[:, None]) ** 2 / (2 * self.sigma_loss_err)) * interval_list
                sigma_loss = sigma_loss + torch.mean(torch.sum(l, dim=1)) / batch_size * self.sigma_loss_weight

            # calculate signed distance field
            if self.beta_mode == 'learnable':
                beta = torch.exp(self._log_beta).expand(len(interpolated_opacity))
            else:
                beta = torch.Tensor([self.gs_scale]).to(interpolated_opacity.device)
            
            density_threshold = 1
            clamped_densities = interpolated_opacity.clamp(min=1e-16)

            # calculate sdf_estimation_loss
            if self.sdf_estimation_loss_weight > 0:
                sdf_values = beta * (torch.sqrt(-2. * torch.log(clamped_densities)) - \
                    np.sqrt(-2. * np.log(min(density_threshold, 1.))))
                sdf_estimation_loss = sdf_estimation_loss + \
                    (sdf_values - sdf_estimation.abs()).abs().mean() * self.sdf_estimation_loss_weight

        return sigma_loss, sdf_estimation_loss
    
    def sample_ray(self, rays_o, rays_d):
        '''Sample query points on rays'''
        rng = self.rng
        rng = rng.repeat(rays_d.shape[-2], 1)
        rng += torch.rand_like(rng[:, [0]])
        Zval = self.stepsize * self.voxel_size * rng
        rays_pts = rays_o[..., None, :] + rays_d[..., None, :] * Zval[..., None]
        rays_pts_depth = (rays_o[..., None, :] - rays_pts).norm(dim=-1)
        mask_outbbox = ((self.xyz_min > rays_pts) | (rays_pts > self.xyz_max)).any(dim=-1)

        return rays_pts, mask_outbbox, Zval, rays_pts_depth
    
    def compute_semantic_loss_flatten(self, sem_est, sem_gt):
        '''
        Args:
            sem_est: N, C
            sem_gt: N
        '''
        if self.contrastive:
            sem_est = torch_scatter.scatter_mean(sem_est, sem_gt, 0)
            sem_gt = torch_scatter.scatter_mean(sem_gt, sem_gt, 0)
            loss = F.cross_entropy(sem_est, sem_gt.long())
        else:
            loss = F.cross_entropy(sem_est, sem_gt.long(), ignore_index=-100)

        return loss

    def compute_semantic_loss(self, sem_est, sem_gt, ignore_index=-100):
        '''
        Args:
            sem_est: B, N, C, H, W, predicted unnormalized logits
            sem_gt: B, N, H, W
        '''
        B, N, C, H, W = sem_est.shape
        sem_est = sem_est.view(B * N, -1, H, W)
        sem_gt = sem_gt.view(B * N, H, W)
        loss = F.cross_entropy(sem_est, sem_gt.long(), ignore_index=ignore_index)

        return loss


@HEADS.register_module()
class GaussianSplattingDecoderWithFlow(GaussianSplattingDecoder):
    def __init__(self,
                 max_means_shift=5,
                 **kwargs):
        super().__init__(**kwargs)

        # predict the flow for each Gaussian, we assume the flow is 2D, no z-axis
        self.flow_predictor = nn.Sequential(
            nn.Linear(self.in_channels, 32),
            nn.ReLU(),
            nn.Linear(32, 2),
            nn.Tanh(),
        )
        self.max_means_shift = max_means_shift

        assert self.pred_density, 'The density prediction is required for flow prediction'

    def forward(self, 
                inputs,
                return_gaussians=False,
                suffix='',
                **kwargs):
        intricics, pose_spatial = inputs['intrinsics'], inputs['pose_spatial']
        
        volume_feat = inputs['volume_feat']  # B, X, Y, Z, C
        render_result_dict, gaussians = self.run_render(
            intricics, pose_spatial, volume_feat=volume_feat)
        
        dec_output = {key + suffix: value for key, value in render_result_dict.items()}
        
        if return_gaussians:
            return dec_output, gaussians
        
        return dec_output
    
    def render_forward(self, 
                       inputs,
                       gaussians,
                       suffix='',):
        
        intrinsics = inputs['intrinsics']
        extrinsics = inputs['pose_spatial']

        b, v = intrinsics.shape[:2]
        device = gaussians.means.device
        
        near = torch.ones(b, v).to(device) * self.min_depth
        far = torch.ones(b, v).to(device) * self.max_depth
        background_color = torch.zeros((3), dtype=torch.float32).to(device)
        
        intrinsics = intrinsics[..., :3, :3]
        # normalize the intrinsics
        intrinsics[:, :, 0] /= self.render_w
        intrinsics[:, :, 1] /= self.render_h

        # use the new extrinsics to compute the Gaussian covariances
        covariances = build_covariance(gaussians.scales, gaussians.rotations)
        covariances = rearrange(covariances, "b g i j -> b () g i j")

        c2w_rotations = extrinsics[..., :3, :3]
        c2w_rotations = rearrange(c2w_rotations, "b v i j -> b v () i j")
        covariances = c2w_rotations @ covariances @ c2w_rotations.transpose(-1, -2)
        gaussians.covariances = covariances  # (bs, v, g, i, j)

        # start rendering
        render_results = render_cuda(
            rearrange(extrinsics, "b v i j -> (b v) i j"),
            rearrange(intrinsics, "b v i j -> (b v) i j"),
            rearrange(near, "b v -> (b v)"),
            rearrange(far, "b v -> (b v)"),
            (self.render_h, self.render_w),
            repeat(background_color, "c -> (b v) c", b=b, v=v),
            repeat(gaussians.means, "b g xyz -> (b v) g xyz", v=v),
            rearrange(gaussians.covariances, "b v g i j -> (b v) g i j"),
            rearrange(gaussians.harmonics, "b v g c d_sh -> (b v) g c d_sh"),
            repeat(gaussians.opacities, "b g -> (b v) g", v=v),
            scale_invariant=False,
            use_sh=True,
            feats3D=gaussians.feats,
            pred_flow=True
        )
        
        keys = ["color", "radii", "depth", "alpha", "proj_2D", 
                "conic_2D", "conic_2D_inv", "gs_per_pixel", 
                "weight_per_gs_pixel", "x_mu"]

        render_result_dict = dict(zip(keys, render_results))
        render_result_dict['render_rgb'] = rearrange(
            render_result_dict['color'], "(b v) c h w -> b v c h w", b=b, v=v)
        render_result_dict['render_depth'] = rearrange(
            render_result_dict['depth'], 
            "(b v) c h w -> b v c h w", b=b, v=v).squeeze(2).clamp(self.min_depth, self.max_depth)
        
        dec_output = {key + suffix: value for key, value in render_result_dict.items()}

        return dec_output

    def run_render(self,
                   intrinsics, 
                   extrinsics, 
                   volume_feat):
        b, v = intrinsics.shape[:2]
        device = volume_feat.device
        
        near = torch.ones(b, v).to(device) * self.min_depth
        far = torch.ones(b, v).to(device) * self.max_depth
        background_color = torch.zeros((3), dtype=torch.float32).to(device)
        
        intrinsics = intrinsics[..., :3, :3]
        # normalize the intrinsics
        intrinsics[:, :, 0] /= self.render_w
        intrinsics[:, :, 1] /= self.render_h

        if self.semantic_head:
            semantic_pred = rearrange(semantic_pred, 'b c h w d -> b (h w d) c')
            _feats3D = repeat(semantic_pred, "b g c -> (b v) g c", v=v)
        else:
            _feats3D = None

        if self.num_offsets == 1:
            gaussians = self.predict_gaussian(None,
                                              extrinsics,
                                              volume_feat)
        else:
            gaussians = self.predict_gaussian_v2(None,
                                                 extrinsics,
                                                 volume_feat)
        
        ## predict the flow for each Gaussian
        means_shift = self.flow_predictor(volume_feat)
        means_shift = rearrange(means_shift, 'b h w d c -> b (h w d) c')
        gaussians.means_shift = self.max_means_shift * means_shift
        
        gaussians.feats = _feats3D

        if self.filter_opacities:
            mask = (gaussians.opacities > 0.0)
            # set the opacities to 0.0 if the opacities are less than 0.0
            gaussians.opacities = gaussians.opacities * mask
        
        # start rendering
        render_results = render_cuda(
            rearrange(extrinsics, "b v i j -> (b v) i j"),
            rearrange(intrinsics, "b v i j -> (b v) i j"),
            rearrange(near, "b v -> (b v)"),
            rearrange(far, "b v -> (b v)"),
            (self.render_h, self.render_w),
            repeat(background_color, "c -> (b v) c", b=b, v=v),
            repeat(gaussians.means, "b g xyz -> (b v) g xyz", v=v),
            rearrange(gaussians.covariances, "b v g i j -> (b v) g i j"),
            rearrange(gaussians.harmonics, "b v g c d_sh -> (b v) g c d_sh"),
            repeat(gaussians.opacities, "b g -> (b v) g", v=v),
            scale_invariant=False,
            use_sh=True,
            feats3D=gaussians.feats,
            pred_flow=True
        )

        keys = ["color", "radii", "depth", "alpha", "proj_2D", 
                "conic_2D", "conic_2D_inv", "gs_per_pixel", 
                "weight_per_gs_pixel", "x_mu"]

        render_result_dict = dict(zip(keys, render_results))
        render_result_dict['render_rgb'] = rearrange(
            render_result_dict['color'], "(b v) c h w -> b v c h w", b=b, v=v)
        render_result_dict['render_depth'] = rearrange(
            render_result_dict['depth'], 
            "(b v) c h w -> b v c h w", b=b, v=v).squeeze(2).clamp(self.min_depth, self.max_depth)

        return render_result_dict, gaussians 