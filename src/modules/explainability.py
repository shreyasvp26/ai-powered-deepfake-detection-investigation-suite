"""Dual Grad-CAM++ for DSAN RGB + frequency streams (plan §11)."""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn

from src.attribution.attribution_model import DSANv3
from src.attribution.gradcam_wrapper import DSANGradCAMWrapper


class ExplainabilityModule:
    """Builds spatial + frequency Grad-CAM++ targets from a trained ``DSANv3``."""

    def __init__(self, dsan_model: DSANv3, device: str = "cpu") -> None:
        self.device = device
        dsan_model.eval()
        self.wrapper = DSANGradCAMWrapper(dsan_model).to(device)
        self.dsan = dsan_model

        from pytorch_grad_cam import GradCAMPlusPlus

        rgb_target = self._find_target_layer(self.wrapper.dsan.rgb_stream.backbone)
        self.rgb_cam = GradCAMPlusPlus(model=self.wrapper, target_layers=[rgb_target])

        freq_target = self._freq_target_layer(self.dsan)
        self.freq_cam = GradCAMPlusPlus(model=self.wrapper, target_layers=[freq_target])

    @staticmethod
    def _find_target_layer(efficientnet_backbone: nn.Module) -> nn.Module:
        target: nn.Module | None = None
        for _name, module in efficientnet_backbone.named_modules():
            if isinstance(module, nn.Conv2d) and module.kernel_size != (1, 1):
                target = module
        if target is None:
            raise RuntimeError("No spatial Conv2d found in EfficientNet backbone")
        return target

    @staticmethod
    def _freq_target_layer(dsan: DSANv3) -> nn.Module:
        # ``FrequencyStream.backbone`` is ResNet ``children()[:-1]``: ends with layer4 then avgpool.
        bb = dsan.freq_stream.backbone
        layer4 = bb[-2]
        last_block = layer4[-1]
        return last_block.conv2

    def generate_heatmaps(
        self,
        rgb_tensor: torch.Tensor,
        srm_tensor: torch.Tensor,
        target_class: int,
    ) -> tuple[np.ndarray, np.ndarray]:
        from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget

        if srm_tensor.dim() == 3:
            srm_tensor = srm_tensor.unsqueeze(0)
        rgb_tensor = rgb_tensor.to(self.device)
        srm_tensor = srm_tensor.to(self.device)
        targets = [ClassifierOutputTarget(target_class)]

        self.wrapper.set_srm(srm_tensor)
        rgb_cam_output = self.rgb_cam(input_tensor=rgb_tensor, targets=targets)
        rgb_heatmap = np.asarray(rgb_cam_output[0], dtype=np.float32)
        rgb_heatmap = self._norm01(rgb_heatmap)

        self.wrapper.set_srm(srm_tensor)
        freq_cam_output = self.freq_cam(input_tensor=rgb_tensor, targets=targets)
        freq_heatmap = np.asarray(freq_cam_output[0], dtype=np.float32)
        freq_heatmap = self._norm01(freq_heatmap)

        return rgb_heatmap, freq_heatmap

    @staticmethod
    def _norm01(x: np.ndarray) -> np.ndarray:
        x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
        lo, hi = float(x.min()), float(x.max())
        if hi - lo < 1e-8:
            return np.zeros_like(x, dtype=np.float32)
        return (x - lo) / (hi - lo)

    def overlay_heatmap(self, original_frame_rgb: np.ndarray, heatmap: np.ndarray) -> np.ndarray:
        from pytorch_grad_cam.utils.image import show_cam_on_image

        frame_float = original_frame_rgb.astype(np.float64) / 255.0
        return show_cam_on_image(frame_float, heatmap, use_rgb=True)
