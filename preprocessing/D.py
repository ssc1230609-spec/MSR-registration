"""
D：Manual Inspection
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Medical Image Artifact Detection Tool

Supports:
- CT: metal / motion / ring / truncation
- MRI: motion / bias field / noise
- Visualization + JSON report + summary

NOTE:
This tool is for preliminary screening.
Manual inspection is still required.

Author: zbh
"""

import os
import json
import argparse
import logging
import traceback
from pathlib import Path
from datetime import datetime
from typing import Dict, Any

import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt
from scipy import ndimage
from scipy.fft import fftn, fftshift
from tqdm import tqdm


# =========================
# Logging
# =========================
logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] %(levelname)s: %(message)s'
)
logger = logging.getLogger(__name__)


# =========================
# JSON Encoder
# =========================
class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)


# =========================
# Detector
# =========================
class ArtifactDetector:

    def __init__(self, modality: str = 'CT'):
        self.modality = modality

    # ---------- IO ----------
    def load_image(self, path: str):
        nii = nib.load(path)
        return nii.get_fdata(), nii

    # ---------- Core Checks ----------
    def check_intensity(self, data: np.ndarray) -> Dict:
        q1, q99 = np.percentile(data, [1, 99])
        std = np.std(data)

        return {
            "min": float(np.min(data)),
            "max": float(np.max(data)),
            "mean": float(np.mean(data)),
            "std": float(std),
            "has_outliers": bool(
                (np.min(data) < q1 - 3 * std) or
                (np.max(data) > q99 + 3 * std)
            )
        }

    def detect_metal(self, data):
        if self.modality != "CT":
            return {"applicable": False}

        ratio = np.mean(data > 3000)
        return {
            "applicable": True,
            "suspected": bool(ratio > 1e-4),
            "ratio": float(ratio)
        }

    def detect_motion(self, data):
        slice_ = data[:, :, data.shape[2] // 2]
        fft_data = np.abs(fftshift(fftn(slice_)))

        energy = np.sum(fft_data**2)
        center = np.array(fft_data.shape) // 2

        y, x = np.ogrid[:fft_data.shape[0], :fft_data.shape[1]]
        dist = np.sqrt((x-center[1])**2 + (y-center[0])**2)

        high = fft_data[dist > min(fft_data.shape)*0.3]
        ratio = np.sum(high**2) / (energy + 1e-10)

        return {
            "suspected": bool(ratio > 0.15),
            "ratio": float(ratio)
        }

    def detect_bias(self, data):
        if self.modality != "MRI":
            return {"applicable": False}

        slice_ = data[:, :, data.shape[2] // 2]
        smooth = ndimage.gaussian_filter(slice_, sigma=20)

        var = np.std(smooth) / (np.mean(smooth) + 1e-10)

        return {
            "applicable": True,
            "suspected": bool(var > 0.3),
            "variation": float(var)
        }

    def detect_noise(self, data):
        corner = min(data.shape) // 10
        noise = np.std(data[:corner, :corner, :corner])
        signal = np.mean(np.abs(data))

        snr = signal / (noise + 1e-10)

        return {
            "snr": float(snr),
            "quality": "poor" if snr < 10 else ("fair" if snr < 20 else "good")
        }

    def detect_truncation(self, data):
        edges = [
            data[0,:,:], data[-1,:,:],
            data[:,0,:], data[:,-1,:],
            data[:,:,0], data[:,:,-1]
        ]

        flags = [np.mean(np.abs(e)) > np.mean(np.abs(data))*0.1 for e in edges]

        return {"suspected": any(flags)}

    # ---------- Visualization ----------
    def visualize(self, data, save_path):
        fig, axes = plt.subplots(1, 3, figsize=(12, 4))

        axes[0].imshow(data[:, :, data.shape[2]//2].T, cmap='gray', origin='lower')
        axes[0].set_title("Axial")

        axes[1].imshow(data[data.shape[0]//2, :, :].T, cmap='gray', origin='lower')
        axes[1].set_title("Sagittal")

        axes[2].imshow(data[:, data.shape[1]//2, :].T, cmap='gray', origin='lower')
        axes[2].set_title("Coronal")

        for ax in axes:
            ax.axis("off")

        plt.tight_layout()
        plt.savefig(save_path, dpi=150)
        plt.close()

    # ---------- Full Analysis ----------
    def analyze(self, path: str, out_dir: str = None) -> Dict[str, Any]:

        logger.info(f"Analyzing {path}")

        data, nii = self.load_image(path)

        result = {
            "file": path,
            "modality": self.modality,
            "shape": data.shape,
            "spacing": nii.header.get_zooms()[:3],
            "intensity": self.check_intensity(data),
            "metal": self.detect_metal(data),
            "motion": self.detect_motion(data),
            "bias": self.detect_bias(data),
            "noise": self.detect_noise(data),
            "truncation": self.detect_truncation(data)
        }

        if out_dir:
            os.makedirs(out_dir, exist_ok=True)
            vis_path = os.path.join(out_dir, Path(path).stem + ".png")
            self.visualize(data, vis_path)
            result["visualization"] = vis_path

        return result


# =========================
# Batch
# =========================
def analyze_dir(ct_dir, mri_dir, out_dir):

    os.makedirs(out_dir, exist_ok=True)

    results = {
        "time": datetime.now().isoformat(),
        "ct": [],
        "mri": []
    }

    # CT
    if ct_dir and os.path.exists(ct_dir):
        detector = ArtifactDetector("CT")
        for f in tqdm(sorted(Path(ct_dir).glob("*.nii.gz")), desc="CT"):
            try:
                results["ct"].append(detector.analyze(str(f), out_dir))
            except Exception as e:
                logger.error(e)

    # MRI
    if mri_dir and os.path.exists(mri_dir):
        detector = ArtifactDetector("MRI")
        for f in tqdm(sorted(Path(mri_dir).glob("*.nii.gz")), desc="MRI"):
            try:
                results["mri"].append(detector.analyze(str(f), out_dir))
            except Exception as e:
                logger.error(e)

    # Save
    report = os.path.join(out_dir, "report.json")
    with open(report, "w") as f:
        json.dump(results, f, indent=2, cls=NumpyEncoder)

    logger.info(f"Saved report: {report}")


# =========================
# CLI
# =========================
def main():
    parser = argparse.ArgumentParser(
        description="Artifact Detection Tool"
    )

    parser.add_argument("--ct_dir", type=str)
    parser.add_argument("--mri_dir", type=str)
    parser.add_argument("--output_dir", type=str, default="./artifact_results")

    parser.add_argument("--single", type=str)
    parser.add_argument("--modality", type=str, choices=["CT", "MRI"], default="CT")

    args = parser.parse_args()

    if args.single:
        det = ArtifactDetector(args.modality)
        result = det.analyze(args.single, args.output_dir)
        print(json.dumps(result, indent=2, cls=NumpyEncoder))
        return

    analyze_dir(args.ct_dir, args.mri_dir, args.output_dir)


if __name__ == "__main__":
    main()
