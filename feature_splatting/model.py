import time
from dataclasses import dataclass, field
from typing import Dict, List, Literal, Optional, Tuple, Type, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from nerfstudio.models.splatfacto import SplatfactoModel, SplatfactoModelConfig, get_viewmat
from nerfstudio.cameras.cameras import Cameras
from nerfstudio.utils.rich_utils import CONSOLE
from nerfstudio.viewer.server.viewer_elements import (
    ViewerButton,
    ViewerNumber,
    ViewerText,
    ViewerCheckbox,
    ViewerSlider,
    ViewerVec3,
)
from nerfstudio.data.scene_box import OrientedBox

# Feature splatting functions
from torch.nn import Parameter
from feature_splatting.utils import (
    ViewerUtils,
    apply_pca_colormap_return_proj,
    two_layer_mlp,
    clip_text_encoder,
    compute_similarity,
    cluster_instance,
    estimate_ground,
    get_ground_bbox_min_max,
    gaussian_editor
)
try:
    from gsplat.cuda_legacy._torch_impl import quat_to_rotmat
    from gsplat.rendering import rasterization
except ImportError:
    print("Please install gsplat>=1.0.0")

@dataclass
class FeatureSplattingModelConfig(SplatfactoModelConfig):
    """Note: make sure to use naming that doesn't conflict with NerfactoModelConfig"""

    _target: Type = field(default_factory=lambda: FeatureSplattingModel)
    # Compute SHs in python
    python_compute_sh: bool = False
    # Weighing for the overall feature loss
    feat_loss_weight: float = 1e-3
    feat_aux_loss_weight: float = 0.1
    # Latent dimension for the feature field
    # TODO(roger): this feat_dim has to add up depth/color to a number that can be rasterized without padding
    # https://github.com/nerfstudio-project/gsplat/blob/main/gsplat/cuda/_wrapper.py#L431
    # gsplat's N-D implementation seems to have some bugs that cause padded tensors to have memory issues
    # we can create a PR to fix this.
    feat_latent_dim: int = 13
    # Feature Field MLP Head
    mlp_hidden_dim: int = 64

def cosine_loss(network_output, gt):
    assert network_output.shape == gt.shape
    return (1 - F.cosine_similarity(network_output, gt, dim=0)).mean()

class FeatureSplattingModel(SplatfactoModel):
    config: FeatureSplattingModelConfig

    def populate_modules(self):
        super().populate_modules()
        # Sanity check
        if self.config.python_compute_sh:
            raise NotImplementedError("Not implemented yet")
        if self.config.sh_degree > 0:
            assert self.config.python_compute_sh, "SH computation is only supported in python"
        else:
            assert not self.config.python_compute_sh, "SHs python compute flag should not be used with 0 SH degree"
        
        # Initialize per-Gaussian features
        distill_features = torch.nn.Parameter(torch.zeros((self.means.shape[0], self.config.feat_latent_dim)))
        self.gauss_params["distill_features"] = distill_features
        self.main_feature_name = self.kwargs["metadata"]["main_feature_name"]
        self.main_feature_shape_chw = self.kwargs["metadata"]["feature_dim_dict"][self.main_feature_name]

        # Initialize the multi-head feature MLP
        self.feature_mlp = two_layer_mlp(self.config.feat_latent_dim,
                                         self.config.mlp_hidden_dim,
                                         self.kwargs["metadata"]["feature_dim_dict"])
        
        # Visualization utils
        self.maybe_populate_text_encoder()
        self.setup_gui()

        self.gaussian_editor = gaussian_editor()
    
    def maybe_populate_text_encoder(self):
        if "clip_model_name" in self.kwargs["metadata"]:
            assert "clip" in self.main_feature_name.lower(), "CLIP model name should only be used with CLIP features"
            self.clip_text_encoder = clip_text_encoder(self.kwargs["metadata"]["clip_model_name"], self.kwargs["device"])
            self.text_encoding_func = self.clip_text_encoder.get_text_token
        else:
            self.text_encoding_func = None
    
    def setup_gui(self):
        self.viewer_utils = ViewerUtils(self.text_encoding_func)
        # Note: the GUI elements are shown based on alphabetical variable names
        self.btn_refresh_pca = ViewerButton("Refresh PCA Projection", cb_hook=lambda _: self.viewer_utils.reset_pca_proj())
        if "clip" in self.main_feature_name.lower():
            self.hint_text = ViewerText(name="Note:", disabled=True, default_value="Use , to separate labels")
            self.lang_1_pos_text = ViewerText(
                name="Positive Text Queries",
                default_value="",
                cb_hook=lambda elem: self.viewer_utils.update_text_embedding('positive', elem.value),
            )
            self.lang_2_neg_text = ViewerText(
                name="Negative Text Queries",
                default_value="object",
                cb_hook=lambda elem: self.viewer_utils.update_text_embedding('negative', elem.value),
            )
            # call the callback function with the default value
            self.viewer_utils.update_text_embedding('negative', self.lang_2_neg_text.default_value)
            self.lang_ground_text = ViewerText(
                name="Ground Text Queries",
                default_value="floor",
                cb_hook=lambda elem: self.viewer_utils.update_text_embedding('ground', elem.value),
            )
            self.viewer_utils.update_text_embedding('ground', self.lang_ground_text.default_value)
            self.softmax_temp = ViewerNumber(
                name="Softmax temperature",
                default_value=self.viewer_utils.softmax_temp,
                cb_hook=lambda elem: self.viewer_utils.update_softmax_temp(elem.value),
            )
            # ===== Start Editing utility =====
            self.edit_checkbox = ViewerCheckbox("Enter Editing Mode", default_value=False, cb_hook=lambda _: self.start_editing())
            # Ground estimation
            self.estimate_ground_btn = ViewerButton("Estimate Ground", cb_hook=lambda _: self.estimate_ground(), disabled=True, visible=False)
            # Main object segmentation
            self.segment_main_obj_btn = ViewerButton("Segment main obj", cb_hook=lambda _: self.segment_positive_obj(), disabled=True, visible=False)
            self.bbox_min_offset_vec = ViewerVec3("BBox Min", default_value=(0, 0, 0), disabled=True, visible=False)
            self.bbox_max_offset_vec = ViewerVec3("BBox Max", default_value=(0, 0, 0), disabled=True, visible=False)
            self.main_obj_only_checkbox = ViewerCheckbox("View main object only", default_value=True, disabled=True, visible=False)
            # Basic editing
            self.translation_vec = ViewerVec3("Translation", default_value=(0, 0, 0), disabled=True, visible=False)
            self.yaw_rotation = ViewerNumber("Yaw-only Rotation (deg)", default_value=0., disabled=True, visible=False)
            # Physics simulation
            self.physics_sim_checkbox = ViewerCheckbox("Physics Simulation", default_value=False, disabled=True, visible=False)
            self.physics_sim_step_btn = ViewerButton("Physics Simulation Step", disabled=True, visible=False, cb_hook=lambda _: self.physics_sim_step())
    
    def physics_sim_step(self):
        # It's just a placeholder now. NS needs some user interaction to send rendering requests.
        # So I make a button that does nothing but to trigger rendering.
        pass
    
    def estimate_ground(self):
        selected_obj_idx, sample_idx = self.segment_gaussian('ground', use_canonical=True, threshold=0.5)
        ground_means_xyz = self.means[sample_idx].detach().cpu().numpy()[selected_obj_idx]
        self.ground_R, self.ground_T, ground_inliers = estimate_ground(ground_means_xyz)
        self.gaussian_editor.register_ground_transform(self.ground_R, self.ground_T)

        # Enable next step
        self.segment_main_obj_btn.set_disabled(False)
        self.segment_main_obj_btn.set_visible(True)
    
    def start_editing(self):
        self.estimate_ground_btn.set_disabled(False)
        self.estimate_ground_btn.set_visible(True)

    def segment_positive_obj(self):
        selected_obj_idx, sample_idx = self.segment_gaussian('positive', use_canonical=False)

        all_xyz = self.means.detach().cpu().numpy()
        selected_xyz = all_xyz[sample_idx]

        selected_obj_idx = cluster_instance(selected_xyz, selected_obj_idx)

        # Get the boolean flag of selected particles (of all particles)
        subset_idx = np.zeros(self.means.shape[0], dtype=bool)
        subset_idx[sample_idx[selected_obj_idx]] = True

        ground_min, ground_max = get_ground_bbox_min_max(all_xyz, subset_idx, self.ground_R, self.ground_T)

        self.gaussian_editor.register_object_minimax(ground_min, ground_max)

        # Enable bbox editing
        self.bbox_min_offset_vec.set_disabled(False)
        self.bbox_min_offset_vec.set_visible(True)
        self.bbox_max_offset_vec.set_disabled(False)
        self.bbox_max_offset_vec.set_visible(True)
        self.main_obj_only_checkbox.set_disabled(False)
        self.main_obj_only_checkbox.set_visible(True)

        # Enable basic editing utilities
        self.translation_vec.set_disabled(False)
        self.translation_vec.set_visible(True)
        self.yaw_rotation.set_disabled(False)
        self.yaw_rotation.set_visible(True)

        # Enable physics simulation
        self.physics_sim_checkbox.set_disabled(False)
        self.physics_sim_checkbox.set_visible(True)
        self.physics_sim_step_btn.set_disabled(False)
        self.physics_sim_step_btn.set_visible(True)
    
    def segment_gaussian(self, field_name : str, use_canonical : bool, sample_size : Optional[int] = 2**15, threshold : Optional[float] = 0.5):
        if "clip" not in self.main_feature_name.lower():
            return
        if sample_size is not None:
            sample_size = min(2**15, self.means.shape[0])
            sample_idx = torch.randperm(self.means.shape[0])[:sample_size]
            sampled_features = self.distill_features[sample_idx]
        else:
            sample_idx = torch.arange(self.means.shape[0])
            sampled_features = self.distill_features
        clip_feature_nc = self.feature_mlp.per_gaussian_forward(sampled_features)[self.main_feature_name]
        clip_feature_nc /= clip_feature_nc.norm(dim=1, keepdim=True)
        clip_feature_cn = clip_feature_nc.permute(1, 0)

        # Use paired softmax method as described in the paper with positive and negative texts
        if not use_canonical and self.viewer_utils.is_embed_valid('negative'):
            neg_embedding = self.viewer_utils.get_text_embed('negative')
        else:
            neg_embedding = self.viewer_utils.get_text_embed('canonical')
        text_embs = torch.cat([self.viewer_utils.get_text_embed(field_name), neg_embedding], dim=0)
        raw_sims = torch.einsum("cm,nc->nm", clip_feature_cn, text_embs)
        pos_sim = compute_similarity(raw_sims, self.viewer_utils.softmax_temp, self.viewer_utils.get_embed_shape(field_name)[0])

        # pos_sim -= pos_sim.min()
        # pos_sim /= pos_sim.max()

        selected_obj_idx = (pos_sim > threshold).cpu().numpy()

        return selected_obj_idx, sample_idx

    def get_outputs(self, camera: Cameras) -> Dict[str, Union[torch.Tensor, List]]:
        """Takes in a camera and returns a dictionary of outputs.

        Args:
            camera: The camera(s) for which output images are rendered. It should have
            all the needed information to compute the outputs.

        Returns:
            Outputs of model. (ie. rendered colors)
        """
        if not isinstance(camera, Cameras):
            print("Called get_outputs with not a camera")
            return {}

        if self.training:
            assert camera.shape[0] == 1, "Only one camera at a time"
            optimized_camera_to_world = self.camera_optimizer.apply_to_camera(camera)
        else:
            optimized_camera_to_world = camera.camera_to_worlds

        # cropping
        if self.crop_box is not None and not self.training:
            crop_ids = self.crop_box.within(self.means).squeeze()
            if crop_ids.sum() == 0:
                return self.get_empty_outputs(
                    int(camera.width.item()), int(camera.height.item()), self.background_color
                )
        else:
            crop_ids = None

        if crop_ids is not None:
            opacities_crop = self.opacities[crop_ids]
            means_crop = self.means[crop_ids]
            features_dc_crop = self.features_dc[crop_ids]
            features_rest_crop = self.features_rest[crop_ids]
            scales_crop = self.scales[crop_ids]
            quats_crop = self.quats[crop_ids]
            distill_features_crop = self.distill_features[crop_ids]
        else:
            opacities_crop = self.opacities
            means_crop = self.means
            features_dc_crop = self.features_dc
            features_rest_crop = self.features_rest
            scales_crop = self.scales
            quats_crop = self.quats
            distill_features_crop = self.distill_features

        # features_dc_crop.shape: [N, 3]
        # features_rest_crop.shape: [N, 15, 3]
        colors_crop = torch.cat((features_dc_crop[:, None, :], features_rest_crop), dim=1)
        # colors_crop.shape: [N, 16, 3]

        BLOCK_WIDTH = 16  # this controls the tile size of rasterization, 16 is a good default
        camera_scale_fac = self._get_downscale_factor()
        camera.rescale_output_resolution(1 / camera_scale_fac)
        viewmat = get_viewmat(optimized_camera_to_world)
        K = camera.get_intrinsics_matrices().cuda()
        W, H = int(camera.width.item()), int(camera.height.item())
        self.last_size = (H, W)
        camera.rescale_output_resolution(camera_scale_fac)  # type: ignore

        # apply the compensation of screen space blurring to gaussians
        if self.config.rasterize_mode not in ["antialiased", "classic"]:
            raise ValueError("Unknown rasterize_mode: %s", self.config.rasterize_mode)

        if self.config.output_depth_during_training or not self.training:
            # Actually render RGB, features, and depth, but can't use RGB+FEAT+ED because we hack gsplat
            render_mode = "RGB+ED"
        else:
            render_mode = "RGB"

        if self.config.sh_degree > 0:
            sh_degree_to_use = min(self.step // self.config.sh_degree_interval, self.config.sh_degree)
            assert self.config.python_compute_sh, "SH computation is only supported in python"
            raise NotImplementedError("Python SHs computation not implemented yet")
            sh_degree_to_use = None
        else:
            colors_crop = torch.sigmoid(colors_crop).squeeze(1)  # [N, 1, 3] -> [N, 3]
            fused_render_properties = torch.cat((colors_crop, distill_features_crop), dim=1)
            sh_degree_to_use = None

        render, alpha, info = rasterization(
            means=means_crop,
            quats=quats_crop / quats_crop.norm(dim=-1, keepdim=True),
            scales=torch.exp(scales_crop),
            opacities=torch.sigmoid(opacities_crop).squeeze(-1),
            colors=fused_render_properties,
            viewmats=viewmat,  # [1, 4, 4]
            Ks=K,  # [1, 3, 3]
            width=W,
            height=H,
            tile_size=BLOCK_WIDTH,
            packed=False,
            near_plane=0.01,
            far_plane=1e10,
            render_mode=render_mode,
            sh_degree=sh_degree_to_use,
            sparse_grad=False,
            absgrad=True,
            rasterize_mode=self.config.rasterize_mode,
            # set some threshold to disregrad small gaussians for faster rendering.
            # radius_clip=3.0,
        )

        if self.training and info["means2d"].requires_grad:
            info["means2d"].retain_grad()
        self.xys = info["means2d"]  # [1, N, 2]
        self.radii = info["radii"][0]  # [N]
        alpha = alpha[:, ...]

        background = self._get_background_color()
        rgb = render[:, ..., :3] + (1 - alpha) * background
        rgb = torch.clamp(rgb, 0.0, 1.0)

        if render_mode == "RGB+ED":
            assert render.shape[3] == 3 + self.config.feat_latent_dim + 1
            depth_im = render[:, ..., -1:]
            depth_im = torch.where(alpha > 0, depth_im, depth_im.detach().max()).squeeze(0)
        else:
            assert render.shape[3] == 3 + self.config.feat_latent_dim
            depth_im = None

        if background.shape[0] == 3 and not self.training:
            background = background.expand(H, W, 3)
        
        feature = render[:, ..., 3:3 + self.config.feat_latent_dim]

        return {
            "rgb": rgb.squeeze(0),  # type: ignore
            "depth": depth_im,  # type: ignore
            "accumulation": alpha.squeeze(0),  # type: ignore
            "background": background,  # type: ignore,
            "feature": feature.squeeze(0),  # type: ignore
        }  # type: ignore

    def decode_features(self, features_hwc: torch.Tensor, resize_factor: float = 1.) -> Dict[str, torch.Tensor]:
        # Decode features
        feature_chw = features_hwc.permute(2, 0, 1)
        feature_shape_hw = (int(self.main_feature_shape_chw[1] * resize_factor), int(self.main_feature_shape_chw[2] * resize_factor))
        rendered_feat = F.interpolate(feature_chw.unsqueeze(0), size=feature_shape_hw, mode="bilinear", align_corners=False)
        rendered_feat_dict = self.feature_mlp(rendered_feat)
        # Rest of the features
        for key, feat_shape_chw in self.kwargs["metadata"]["feature_dim_dict"].items():
            if key != self.main_feature_name:
                rendered_feat_dict[key] = F.interpolate(rendered_feat_dict[key], size=feat_shape_chw[1:], mode="bilinear", align_corners=False)
            rendered_feat_dict[key] = rendered_feat_dict[key].squeeze(0)
        return rendered_feat_dict
    
    def get_loss_dict(self, outputs, batch, metrics_dict=None) -> Dict[str, torch.Tensor]:
        # Splatfacto computes the loss for the rgb image
        loss_dict = super().get_loss_dict(outputs, batch, metrics_dict)
        for k in batch['feature_dict']:
            batch['feature_dict'][k] = batch['feature_dict'][k].to(self.device)
        decoded_feature_dict = self.decode_features(outputs["feature"])
        feature_loss = torch.tensor(0.0, device=self.device)
        for key, target_feat in batch['feature_dict'].items():
            cur_loss_weight = 1.0 if key == self.main_feature_name else self.config.feat_aux_loss_weight
            ignore_feat_mask = (torch.sum(target_feat == 0, dim=0) == target_feat.shape[0])
            target_feat[:, ignore_feat_mask] = decoded_feature_dict[key][:, ignore_feat_mask]
            feature_loss += cosine_loss(decoded_feature_dict[key], target_feat) * cur_loss_weight
        loss_dict["feature_loss"] = self.config.feat_loss_weight * feature_loss
        return loss_dict
    
    @torch.no_grad()
    def get_outputs_for_camera(self, camera: Cameras, obb_box: Optional[OrientedBox] = None) -> Dict[str, torch.Tensor]:
        """This function is not called during training, but used for visualization in browser. So we can use it to
        add visualization not needed during training.
        """
        editing_dict = self.gaussian_editor.prepare_editing_dict(self.translation_vec.value, self.yaw_rotation.value, self.physics_sim_checkbox.value)
        if self.edit_checkbox.value:
            # Editing mode
            self.gaussian_editor.pre_rendering_process(self.means, self.opacities, self.scales, self.quats,
                                                       editing_dict=editing_dict,
                                                       min_offset=torch.tensor(self.bbox_min_offset_vec.value).float().cuda() / 10.0,
                                                       max_offset=torch.tensor(self.bbox_max_offset_vec.value).float().cuda() / 10.0,
                                                       view_main_obj_only=self.main_obj_only_checkbox.value)
        outs = super().get_outputs_for_camera(camera, obb_box)
        if self.edit_checkbox.value:
            self.gaussian_editor.post_rendering_process(self.means, self.opacities, self.quats, self.scales)
        if self.physics_sim_checkbox.value:
            # turn off feature rendering during physics sim for speed
            return outs
        # Consistent pca that does not flicker
        outs["consistent_latent_pca"], self.viewer_utils.pca_proj, *_ = apply_pca_colormap_return_proj(
            outs["feature"], self.viewer_utils.pca_proj
        )
        # TODO(roger): this resize factor affects the resolution of similarity map. Maybe we should use a fixed size?
        decoded_feature_dict = self.decode_features(outs["feature"], resize_factor=8)
        if "clip" in self.main_feature_name.lower() and self.viewer_utils.is_embed_valid('positive'):
            clip_features = decoded_feature_dict[self.main_feature_name]
            clip_features /= clip_features.norm(dim=0, keepdim=True)

            # Use paired softmax method as described in the paper with positive and negative texts
            if self.viewer_utils.is_embed_valid('negative'):
                neg_embedding = self.viewer_utils.get_text_embed('negative')
            else:
                neg_embedding = self.viewer_utils.get_text_embed('canonical')
            text_embs = torch.cat([self.viewer_utils.get_text_embed('positive'), neg_embedding], dim=0)
            raw_sims = torch.einsum("chw,nc->nhw", clip_features, text_embs)
            sim_shape_hw = raw_sims.shape[1:]

            raw_sims = raw_sims.reshape(raw_sims.shape[0], -1)
            pos_sim = compute_similarity(raw_sims, self.viewer_utils.softmax_temp, self.viewer_utils.get_embed_shape('positive')[0])
            outs["similarity"] = pos_sim.reshape(sim_shape_hw + (1,)) # H, W, 1
            
            # Upsample heatmap to match size of RGB image
            # It's a bit slow since we do it on full resolution; but interpolation seems to have aliasing issues
            assert outs["similarity"].shape[2] == 1
            if outs["similarity"].shape[:2] != outs["rgb"].shape[:2]:
                out_sim = outs["similarity"][:, :, 0]  # H, W
                out_sim = out_sim[None, None, ...]  # 1, 1, H, W
                outs["similarity"] = F.interpolate(out_sim, size=outs["rgb"].shape[:2], mode="bilinear", align_corners=False).squeeze()
                outs["similarity"] = outs["similarity"][:, :, None]
        return outs
    
    # ===== Utils functions for managing the gaussians =====

    @property
    def distill_features(self):
        return self.gauss_params["distill_features"]
    
    def load_state_dict(self, dict, **kwargs):  # type: ignore
        # resize the parameters to match the new number of points
        self.step = 30000
        if "means" in dict:
            # For backwards compatibility, we remap the names of parameters from
            # means->gauss_params.means since old checkpoints have that format
            for p in self.gauss_params.keys():
                dict[f"gauss_params.{p}"] = dict[p]
        newp = dict["gauss_params.means"].shape[0]
        for name, param in self.gauss_params.items():
            old_shape = param.shape
            new_shape = (newp,) + old_shape[1:]
            self.gauss_params[name] = torch.nn.Parameter(torch.zeros(new_shape, device=self.device))
        super().load_state_dict(dict, **kwargs)
    
    def split_gaussians(self, split_mask, samps):
        """
        This function splits gaussians that are too large
        """
        n_splits = split_mask.sum().item()
        CONSOLE.log(f"Splitting {split_mask.sum().item()/self.num_points} gaussians: {n_splits}/{self.num_points}")
        centered_samples = torch.randn((samps * n_splits, 3), device=self.device)  # Nx3 of axis-aligned scales
        scaled_samples = (
            torch.exp(self.scales[split_mask].repeat(samps, 1)) * centered_samples
        )  # how these scales are rotated
        quats = self.quats[split_mask] / self.quats[split_mask].norm(dim=-1, keepdim=True)  # normalize them first
        rots = quat_to_rotmat(quats.repeat(samps, 1))  # how these scales are rotated
        rotated_samples = torch.bmm(rots, scaled_samples[..., None]).squeeze()
        new_means = rotated_samples + self.means[split_mask].repeat(samps, 1)
        # step 2, sample new colors
        new_features_dc = self.features_dc[split_mask].repeat(samps, 1)
        new_features_rest = self.features_rest[split_mask].repeat(samps, 1, 1)
        # step 3, sample new opacities
        new_opacities = self.opacities[split_mask].repeat(samps, 1)
        # step 4, sample new scales
        size_fac = 1.6
        new_scales = torch.log(torch.exp(self.scales[split_mask]) / size_fac).repeat(samps, 1)
        self.scales[split_mask] = torch.log(torch.exp(self.scales[split_mask]) / size_fac)
        # step 5, sample new quats
        new_quats = self.quats[split_mask].repeat(samps, 1)
        # step 6 (RQ, July 2024), sample new distill_features
        new_distill_features = self.distill_features[split_mask].repeat(samps, 1)
        out = {
            "means": new_means,
            "features_dc": new_features_dc,
            "features_rest": new_features_rest,
            "opacities": new_opacities,
            "scales": new_scales,
            "quats": new_quats,
            "distill_features": new_distill_features,
        }
        for name, param in self.gauss_params.items():
            if name not in out:
                out[name] = param[split_mask].repeat(samps, 1)
        return out

    def get_gaussian_param_groups(self) -> Dict[str, List[Parameter]]:
        # Here we explicitly use the means, scales as parameters so that the user can override this function and
        # specify more if they want to add more optimizable params to gaussians.
        return {
            name: [self.gauss_params[name]]
            for name in self.gauss_params.keys()
        }
    
    def get_param_groups(self) -> Dict[str, List[Parameter]]:
        # Gather Gaussian-related parameters
        # The distill_features parameter is added via the get_gaussian_param_groups method
        param_groups = super().get_param_groups()
        param_groups["feature_mlp"] = list(self.feature_mlp.parameters())
        return param_groups
    
    def get_metrics_dict(self, outputs, batch) -> Dict[str, torch.Tensor]:
        """Compute and returns metrics.

        Args:
            outputs: the output to compute loss dict to
            batch: ground truth batch corresponding to outputs
        """
        gt_rgb = self.composite_with_background(self.get_gt_img(batch["image"]), outputs["background"])
        metrics_dict = {}
        predicted_rgb = outputs["rgb"]
        metrics_dict["psnr"] = self.psnr(predicted_rgb, gt_rgb)

        metrics_dict["gaussian_count"] = self.num_points

        self.camera_optimizer.get_metrics_dict(metrics_dict)
        return metrics_dict
