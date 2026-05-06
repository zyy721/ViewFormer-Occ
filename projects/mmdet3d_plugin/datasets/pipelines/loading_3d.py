import mmcv
import numpy as np
import os.path as osp
from mmdet.datasets.builder import PIPELINES
import cv2
import torch
from mmcv.image.photometric import imnormalize
from mmcv.parallel import DataContainer as DC
from mmdet.datasets.pipelines import to_tensor
from pyquaternion import Quaternion

# from projects.mmdet3d_plugin.models.dense_heads.vggt.utils.load_fn import load_and_preprocess_images


def rt2mat(translation, quaternion=None, inverse=False, rotation=None):
    R = Quaternion(quaternion).rotation_matrix if rotation is None else rotation
    T = np.array(translation)
    if inverse:
        R = R.T
        T = -R @ T
    mat = np.eye(4)
    mat[:3, :3] = R
    mat[:3, 3] = T
    return mat


@PIPELINES.register_module()
class LoadMultiViewMultiSweepImageFromFiles(object):
    """Load multi channel images from a list of separate channel files.

    Expects results['img_filename'] to be a list of filenames.

    Args:
        to_float32 (bool): Whether to convert the img to float32.
            Defaults to False.
        color_type (str): Color type of the file. Defaults to 'unchanged'.
    """

    def __init__(
        self,
        to_float32=False,
        sweep_num=1,
        random_sweep=False,
        color_type="unchanged",
        file_client_args=dict(backend="disk"),

        use_reliev3R=False,
        use_sc_depth=False,

    ):
        self.to_float32 = to_float32
        self.color_type = color_type
        self.sweep_num = sweep_num
        self.random_sweep = random_sweep
        self.file_client_args = file_client_args.copy()
        self.file_client = None

        self.use_reliev3R = use_reliev3R
        self.use_sc_depth = use_sc_depth

    def _load_image(self, img_filename):
        if self.file_client is None:
            self.file_client = mmcv.FileClient(**self.file_client_args)
        try:
            img_bytes = self.file_client.get(img_filename)
            image = np.frombuffer(img_bytes, dtype=np.uint8)
            image = cv2.imdecode(image, cv2.IMREAD_COLOR)
        except ConnectionError:
            image = mmcv.imread(img_filename, self.color_type)
        if self.to_float32:
            image = image.astype(np.float32)
        return image

    def __call__(self, results):
        """Call function to load multi-view image from files.

        Args:
            results (dict): Result dict containing multi-view image filenames.

        Returns:
            dict: The result dict containing the multi-view image data. \
                Added keys and values are described below.

                - filename (str): Multi-view image filenames.
                - img (np.ndarray): Multi-view image arrays.
                - img_shape (tuple[int]): Shape of multi-view image arrays.
                - ori_shape (tuple[int]): Shape of original image arrays.
                - pad_shape (tuple[int]): Shape of padded image arrays.
                - scale_factor (float): Scale factor.
                - img_norm_cfg (dict): Normalization configuration of images.
        """
        filename = results["img_filename"]
        results["filename"] = filename

        imgs = [self._load_image(name) for name in filename]

        sweeps_paths = results["cam_sweeps_paths"]
        sweeps_ids = results["cam_sweeps_id"]
        sweeps_time = results["cam_sweeps_time"]
        if self.random_sweep:
            random_num = np.random.randint(0, self.sweep_num)
            sweeps_paths = [_sweep[:random_num] for _sweep in sweeps_paths]
            sweeps_ids = [_sweep[:random_num] for _sweep in sweeps_ids]
        else:
            random_num = self.sweep_num

        sweeps_imgs = []
        for cam_idx in range(len(sweeps_paths)):
            sweeps_imgs.extend(
                [imgs[cam_idx]]
                + [self._load_image(name) for name in sweeps_paths[cam_idx]]
            )

        results["sweeps_paths"] = [
            [filename[_idx]] + sweeps_paths[_idx] for _idx in range(len(filename))
        ]
        results["sweeps_ids"] = np.stack([[0] + _id for _id in sweeps_ids], axis=-1)
        results["sweeps_time"] = np.stack(
            [[0] + _time for _time in sweeps_time], axis=-1
        )
        # unravel to list, see `DefaultFormatBundle` in formating.py
        # which will transpose each image separately and then stack into array
        results["img_ori"] = imgs
        results["img"] = sweeps_imgs
        results["img_shape"] = [img.shape for img in sweeps_imgs]
        results["ori_shape"] = [img.shape for img in sweeps_imgs]
        # Set initial values for default meta_keys
        results["pad_shape"] = [img.shape for img in sweeps_imgs]
        results["pad_before_shape"] = [img.shape for img in sweeps_imgs]
        results["scale_factor"] = 1.0
        num_channels = 1 if len(imgs[0].shape) < 3 else imgs[0].shape[2]
        results["img_norm_cfg"] = dict(
            mean=np.zeros(num_channels, dtype=np.float32),
            std=np.ones(num_channels, dtype=np.float32),
            to_rgb=False,
        )

        # add sweep matrix to raw matrix
        results["lidar2img"] = [
            np.stack(
                [
                    results["lidar2img"][_idx],
                    *results["lidar2img_sweeps"][_idx][:random_num],
                ],
                axis=0,
            )
            for _idx in range(len(results["lidar2img"]))
        ]
        results["lidar2cam"] = [
            np.stack(
                [
                    results["lidar2cam"][_idx],
                    *results["lidar2cam_sweeps"][_idx][:random_num],
                ],
                axis=0,
            )
            for _idx in range(len(results["lidar2cam"]))
        ]
        results["cam_intrinsic"] = [
            np.stack(
                [
                    results["cam_intrinsic"][_idx],
                    *results["cam_sweeps_intrinsics"][_idx][:random_num],
                ],
                axis=0,
            )
            for _idx in range(len(results["cam_intrinsic"]))
        ]

        if self.use_reliev3R or self.use_sc_depth:
            results["global2cam"] = [
                np.stack(
                    [
                        results["global2cam"][_idx],
                        *results["global2cam_sweeps"][_idx][:random_num],
                    ],
                    axis=0,
                )
                for _idx in range(len(results["global2cam"]))
            ]
            results.pop("global2cam_sweeps")


        results.pop("lidar2img_sweeps")
        results.pop("lidar2cam_sweeps")
        results.pop("cam_sweeps_intrinsics")


        return results

    def __repr__(self):
        """str: Return a string that describes the module."""
        repr_str = self.__class__.__name__
        repr_str += f"(to_float32={self.to_float32}, "
        repr_str += f"color_type='{self.color_type}')"
        return repr_str


@PIPELINES.register_module()
class PrepapreImageInputs(LoadMultiViewMultiSweepImageFromFiles):
    """Prepare the original mage inputs for the SSL.
    """
    def __init__(self, 
                 input_size, 
                 norm_cfg=None,
                 load_future_img=False,
                 load_prev_img=False,
                 **kwargs):
        super().__init__(**kwargs)

        self.input_size = input_size  # (h, w)
        if norm_cfg is not None:
            self.mean = norm_cfg['mean']
            self.std = norm_cfg['std']
        else:
            self.mean = np.array([0.0, 0.0, 0.0], dtype=np.float32)
            self.std = np.array([255.0, 255.0, 255.0], dtype=np.float32)

        self.load_prev_img = load_prev_img
        self.load_future_img = load_future_img

    def transform_core(self, 
                       img, 
                       img_size, # (h, w)
                       to_rgb=True):
        ## we need [0, 1] images in RGB order
        img = cv2.resize(img, img_size[::-1])
        img = imnormalize(np.array(img), self.mean, self.std, to_rgb)
        return img

    def prepare_data(self, results):
        try:
            # obtain the current lidar to ego transformation
            curr_lidar_to_ego = rt2mat(results['lidar2ego_translation'],
                                    results['lidar2ego_rotation'])
            curr_lidarego_to_global = rt2mat(results['ego2global_translation'],
                                            results['ego2global_rotation'])
        except:
            pass
        
        if 'adjacent' in results and results['adjacent'] is not None:
            source_imgs_list = []
            source_cam_to_tgt_frame_transform = []
            for adj_info in results['adjacent']:
                cam_infos = adj_info["cams"]
                img_filename = [cam_info['data_path'] for cam_info in cam_infos.values()]
                img_adj = [self._load_image(name) for name in img_filename]

                # resize and normalize
                imgs = [
                    self.transform_core(img, self.input_size)
                    for img in img_adj
                ]
                imgs = [img.transpose(2, 0, 1) for img in imgs]
                source_imgs_list.append(np.ascontiguousarray(np.stack(imgs, axis=0)))

                try:
                    # get the adj frame camera to current frame lidar transformation
                    # 1) adj frame lidar to global
                    # 2) adj frame global to current frame ego
                    # obtain the current ego to future ego transformation, please 
                    # note that the 'ego' here is lidar coordinate system.
                    adj_global2lidarego = rt2mat(adj_info['ego2global_translation'],
                                                adj_info['ego2global_rotation'],
                                                inverse=True)
                    adj_lidarego2lidar = rt2mat(adj_info['lidar2ego_translation'],
                                                adj_info['lidar2ego_rotation'],
                                                inverse=True)
                    adj_lidar2cam = np.stack(adj_info['lidar2cam']) # (6, 4, 4)
                    # (4, 4) matrix
                    curr_lidar_to_adj_cam = \
                        adj_lidar2cam @ adj_lidarego2lidar @ adj_global2lidarego @ curr_lidarego_to_global @ curr_lidar_to_ego
                    
                    adj_cam2curr_lidar = np.linalg.inv(curr_lidar_to_adj_cam)
                    source_cam_to_tgt_frame_transform.append(adj_cam2curr_lidar)
                except:
                    pass

            if source_cam_to_tgt_frame_transform:    
                results['source_cam_to_tgt_frame_transform']  = np.stack(source_cam_to_tgt_frame_transform, axis=0)
            results['source_imgs'] = DC(
                to_tensor(np.stack(source_imgs_list, axis=0)), stack=True)
        
        img_ori = results['img_ori']
        ## resize the image
        imgs = [
            self.transform_core(img, self.input_size)
            for img in img_ori
        ]

        # process multiple imgs in single frame
        imgs = [img.transpose(2, 0, 1) for img in imgs]
        imgs = np.ascontiguousarray(np.stack(imgs, axis=0))
        results['target_imgs'] = DC(to_tensor(imgs), stack=True)

        if 'K' in results and results['K'] is not None:
            # process the intrinsic matrix
            if isinstance(results['ori_shape'], tuple):
                ori_shape = list(results['ori_shape'])
            else:
                ori_shape = results['ori_shape'][0]

            origin_h, origin_w = ori_shape[0], ori_shape[1]
            h, w = self.input_size[0], self.input_size[1]
            results['K'][:, 0] *= w / origin_w
            results['K'][:, 1] *= h / origin_h
            results['inv_K'] = torch.pinverse(results['K'])

    def __call__(self, results):
        self.prepare_data(results)

        if self.load_future_img:
            # load the future anchor image
            assert 'future_info' in results, \
                'future_info should be in results'
            future_results = results['future_info']

            filename = future_results["img_filename"]
            imgs = [self._load_image(name) for name in filename]

            future_results = dict()
            future_results["img_ori"] = imgs
            future_results["ori_shape"] = [img.shape for img in imgs]
            future_results["adjacent"] = results.get('adjacent_future', None)
            future_results["K"] = results.get('K_future', None)

            # load the future adjacent image
            self.prepare_data(future_results)

            # update to the results
            if 'source_imgs' in future_results:
                results['source_imgs_future'] = future_results['source_imgs']
            
            results['target_imgs_future'] = future_results['target_imgs']

            if 'inv_K' in future_results:
                results['K_future'] = future_results['K']
                results['inv_K_future'] = future_results['inv_K']
        
        if self.load_prev_img:
            # load the future anchor image
            assert 'prev_info' in results, \
                'prev_info should be in results'
            prev_results = results['prev_info']

            filename = prev_results["img_filename"]
            imgs = [self._load_image(name) for name in filename]

            prev_results = dict()
            prev_results["img_ori"] = imgs
            prev_results["ori_shape"] = [img.shape for img in imgs]
            prev_results["adjacent"] = results.get('adjacent_prev', None)
            prev_results["K"] = results.get('K_prev', None)

            # load the previous adjacent image
            self.prepare_data(prev_results)

            # update to the results
            if 'source_imgs' in prev_results:
                results['source_imgs_prev'] = prev_results['source_imgs']
            
            results['target_imgs_prev'] = prev_results['target_imgs']

            if 'inv_K' in prev_results:
                results['K_prev'] = prev_results['K']
                results['inv_K_prev'] = prev_results['inv_K']

        return results
    

@PIPELINES.register_module()
class PreparePseudoDepth(object):
    def __init__(self, data_dir, target_size):
        self.data_dir = data_dir
        self.target_size = target_size  # (h, w)

    def _load_depth(self, filepath):
        if filepath.endswith('.npz'):
            depth = np.load(filepath)['depth']
        else:
            raise NotImplementedError
        
        return depth

    def __call__(self, results):
        img_filename = results["img_filename"]
        depth_filename = ['/'.join(img_fp.split('/')[-2:]) for img_fp in img_filename]
        depth_fp = [osp.join(self.data_dir, name.replace('.jpg', '.npz')) for name in depth_filename]
        depth_ori = np.stack([self._load_depth(fp) for fp in depth_fp], axis=0)
        depth = torch.from_numpy(depth_ori)

        # resize to the target size
        depth = torch.nn.functional.interpolate(
            depth.unsqueeze(1),
            size=self.target_size,
            mode="bicubic",
            align_corners=False,
        ).squeeze()
        
        results["pseudo_depth"] = DC(depth, stack=True)  # (num_cam, H, W)
        return results


# @PIPELINES.register_module()
# class PrepapreImageForVGGT(object):
#     """Prepare the original mage inputs for the SSL.
#     """
#     def __init__(self, 
#                  **kwargs):
#         super().__init__(**kwargs)

#         self.camera_names = ['CAM_FRONT', 'CAM_FRONT_LEFT', 'CAM_BACK_LEFT', 
#                              'CAM_BACK', 'CAM_BACK_RIGHT', 'CAM_FRONT_RIGHT']

#     def __call__(self, results):

#         # Load and preprocess example images (replace with your own image paths)

#         image_names = []
#         for cam in self.camera_names:
#             cur_img_path = results['cams'][cam]['data_path']
#             image_names.append(cur_img_path)

#         images = load_and_preprocess_images(image_names)
#         results['img_vggt'] = images

#         return results