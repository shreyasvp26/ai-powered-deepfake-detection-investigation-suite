# Environment verification — run on local and remote before development (PROJECT_PLAN_v10 §4.4)
import sys

import cv2
import mediapipe as mp
import streamlit
import timm
import torch
from facenet_pytorch import MTCNN

print(f"Python: {sys.version}")

print(f"PyTorch: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
if hasattr(torch.backends, "mps"):
    mps_ok = torch.backends.mps.is_available()
    print(f"MPS available: {mps_ok}")

# Project treats local Mac as CPU-only for this stack
if torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"
print(f"Using device: {device}")

print(f"MediaPipe: {mp.__version__}")

print(f"OpenCV: {cv2.__version__}")

_ = MTCNN  # ensure facenet-pytorch / MTCNN import works
print("MTCNN: OK")

print(f"timm: {timm.__version__}")

print(f"Streamlit: {streamlit.__version__}")

print("\n--- All dependencies verified ---")
