"""DSAN v3 / v3.1 — dual-stream attribution model, dataset, losses, samplers.

v3 classes (``DSANv3``, ``DSANLoss``, ``DSANDataset``) are preserved for
backwards compatibility and baseline reproducibility. v3.1 (``DSANv31``,
``DSANv31Loss``, ``DSANv31Dataset``, ``MaskDecoder``, SBI, EMA, Mixup) is the
Excellence pass used by ``training/train_attribution_v31.py`` and documented
in ``docs/GPU_EXECUTION_PLAN.md`` §12.
"""

from src.attribution.attribution_model import DSANv3
from src.attribution.attribution_model_v31 import DSANv31
from src.attribution.dataset import DSANDataset
from src.attribution.dataset_v31 import DSANv31Dataset
from src.attribution.ema import ExponentialMovingAverage
from src.attribution.gated_fusion import GatedFusion
from src.attribution.gradcam_wrapper import DSANGradCAMWrapper
from src.attribution.losses import DSANLoss, DSANv31Loss, SupConLoss
from src.attribution.mask_decoder import MaskDecoder
from src.attribution.mixup import mixup_batch, mixup_ce_loss
from src.attribution.samplers import StratifiedBatchSampler
from src.attribution.sbi import SBIConfig, mask_from_ff_annotation, synth_sbi

__all__ = [
    "DSANDataset",
    "DSANGradCAMWrapper",
    "DSANLoss",
    "DSANv3",
    "DSANv31",
    "DSANv31Dataset",
    "DSANv31Loss",
    "ExponentialMovingAverage",
    "GatedFusion",
    "MaskDecoder",
    "SBIConfig",
    "StratifiedBatchSampler",
    "SupConLoss",
    "mask_from_ff_annotation",
    "mixup_batch",
    "mixup_ce_loss",
    "synth_sbi",
]
