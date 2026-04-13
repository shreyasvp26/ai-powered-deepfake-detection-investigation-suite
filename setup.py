"""Editable install for package imports (e.g. from src.modules.spatial import SpatialDetector)."""

from setuptools import find_packages, setup

setup(
    name="deepfake-detection",
    version="1.0.0",
    packages=find_packages(),
    python_requires=">=3.10",
)
