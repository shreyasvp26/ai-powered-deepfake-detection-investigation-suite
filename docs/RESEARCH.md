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
