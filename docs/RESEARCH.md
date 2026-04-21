# Research references

Literature tied to [PROJECT_PLAN.md](PROJECT_PLAN.md) Section 27. Each entry: citation + 2–3 sentences of relevance.

1. **Rossler et al., "FaceForensics++: Learning to Detect Manipulated Facial Images," ICCV 2019** — Defines the FF++ benchmark, manipulation categories, and the XceptionNet baseline used for spatial (binary) detection in this project.

2. **FAME: "A Lightweight Spatio-Temporal Network for Model Attribution of Face-Swap Deepfakes," ESWA 2025** — Frames attribution (which method?) as a first-class task alongside detection; motivates our 4-way attribution evaluation on FF++.

3. **Hao et al., "Fighting Fake News: Two Stream Network for Deepfake Detection via Learnable SRM," IEEE TBIOM 2021** — RGB + learnable noise/residual streams; direct precedent for combining spatial RGB features with forensic high-frequency cues in DSAN.

4. **Frank et al., "Leveraging Frequency Analysis for Deep Fake Image Recognition," ICML 2020** — Spectral artifacts from GAN upsampling; supports the FFT branch in our frequency stream.

5. **SFANet: "Spatial-Frequency Attention Network for Deepfake Detection," 2024** — Recent spatial–frequency design; useful comparison point for dual-stream forensic architectures.

6. **Khosla et al., "Supervised Contrastive Learning," NeurIPS 2020** — SupCon formulation; we adapt temperature and batching (effective batch ~96) for the contrastive head in DSAN training.

7. **DATA: "Multi-disentanglement based contrastive learning for deepfake attribution," 2025** — Contrastive learning for attribution in broader settings; contextualises our supervised contrastive loss on FF++ method labels.

8. **ForensicFlow: "A Tri-Modal Adaptive Network for Robust Deepfake Detection," 2025** — Multi-modal fusion; relates to our gated fusion of RGB and frequency embeddings.

9. **AWARE-NET: Two-tier ensemble framework, 2025** — Strong FF++ detection numbers with augmentation; sets an upper-bound reference for binary detection metrics.

10. **Solanki et al., "In Ictu Oculi: Exposing AI Created Fake Videos by Detecting Eye Blinking," 2018** — Classic blink-based fake detection; cited when explaining why our blink module is deprecated on c23 + low FPS (method is valid, dataset/sampling constraints are not).

11. **He et al., "Momentum Contrast for Unsupervised Visual Representation Learning," CVPR 2020** — MoCo and memory-bank contrastive ideas; optional reference if we need larger effective batches for SupCon later.

12. **Li et al., "Celeb-DF v2: A New Dataset for DeepFake Forensics," CVPR 2020** — Cross-dataset evaluation benchmark; we use a 100-video smoke slice to report the honest FF++ → Celeb-DF generalisation drop.

13. **Dolhansky et al., "The DeepFake Detection Challenge (DFDC) Dataset," 2020** — Facebook's large-scale in-the-wild benchmark; we use the preview subset to report additional cross-dataset robustness.

14. **Selvaraju et al., "Grad-CAM: Visual Explanations from Deep Networks via Gradient-based Localization," ICCV 2017** — Theoretical basis for our Grad-CAM++ implementation targeting the last spatial Conv2d on EfficientNet-B4 and `layer4.conv2` on ResNet-18.

---

## Dropped features — rationale

### Blink detection (former F003)

Originally planned as a third parallel signal. Dropped for V1. Reasons:

- **Temporal variance already covers it.** On FF++ c23, the dominant blink-detection signal (Solanki et al. 2018) is *low blink rate*, which in turn manifests as *low per-frame variance* in the spatial score. `TemporalAnalyzer` already captures this with its `global_variance` + `sign_flip_rate` features.
- **MediaPipe EAR + XGBoost was brittle.** c23 compression and 1 FPS sampling together make EAR traces noisy; blink detection AUC on FF++ c23 is below the project's "signal worth fusing" threshold.
- **Complexity does not pay back.** A separate blink module adds a dependency (MediaPipe on Linux + Mac arm64), a classifier (XGBoost), a training notebook, and three configuration knobs — for a signal the existing fusion already approximates.
- **Literature migrating.** Modern attribution architectures (DSAN, SFANet, ForensicFlow) do not use blink; they rely on spatial + frequency fusion, which we mirror.

The code files (`src/modules/blink.py`, `training/train_blink_classifier.py`, `tests/test_blink.py`, `notebooks/04_blink_detection.ipynb`) were planned but **not created**. `PROJECT_PLAN_v10.md` §8 is retained as historical reference because the plan is explicitly versioned.

### RetinaFace on macOS (former plan §4.2 promise)

Dropped because `insightface` is Linux-oriented and installation on macOS arm64 was unreliable. MTCNN (`facenet-pytorch`) covers both platforms. RetinaFace remains available on the Linux GPU server for batch face extraction.

