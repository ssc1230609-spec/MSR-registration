"""
A： N4 bias field correction
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
N4 Bias Field Correction for MRI (SimpleITK)

Features:
- N4 bias field correction
- Otsu-based automatic mask generation
- Shrink factor acceleration
- Single-case and batch processing
- Logging + CLI support

Author:zbh
"""

import os
import argparse
import logging
import traceback
from pathlib import Path
from typing import Optional, List

import SimpleITK as sitk
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
# Core Function
# =========================
def n4_bias_correction(
    input_path: str,
    output_path: str,
    mask_path: Optional[str] = None,
    shrink_factor: int = 4,
    num_iterations: List[int] = [50, 50, 50, 50]
) -> bool:
    """
    Perform N4 bias field correction on MRI.

    Args:
        input_path: input MRI path
        output_path: output corrected MRI path
        mask_path: optional mask path
        shrink_factor: downsampling factor for speed
        num_iterations: iteration per level

    Returns:
        success flag
    """

    try:
        logger.info(f"Processing: {input_path}")

        # Read image
        image = sitk.ReadImage(input_path, sitk.sitkFloat32)

        # Mask
        if mask_path is None:
            mask_image = sitk.OtsuThreshold(image, 0, 1, 200)
        else:
            mask_image = sitk.ReadImage(mask_path, sitk.sitkUInt8)

        # Shrink for speed
        if shrink_factor > 1:
            shrink = [shrink_factor] * image.GetDimension()
            image_shrunk = sitk.Shrink(image, shrink)
            mask_shrunk = sitk.Shrink(mask_image, shrink)
        else:
            image_shrunk = image
            mask_shrunk = mask_image

        # N4 Corrector
        corrector = sitk.N4BiasFieldCorrectionImageFilter()
        corrector.SetMaximumNumberOfIterations(num_iterations)
        corrector.SetConvergenceThreshold(1e-3)

        logger.info("Running N4 correction...")
        corrected_shrunk = corrector.Execute(image_shrunk, mask_shrunk)

        # Recover full resolution
        if shrink_factor > 1:
            logger.info("Resampling bias field to original resolution...")

            log_bias = corrector.GetLogBiasFieldAsImage(image_shrunk)
            log_bias = sitk.Resample(log_bias, image)

            bias_field = sitk.Exp(log_bias)
            corrected_full = image / bias_field
        else:
            corrected_full = corrected_shrunk

        # Save
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        sitk.WriteImage(corrected_full, output_path)

        logger.info(f"Saved: {output_path}")
        return True

    except Exception as e:
        logger.error(f"Failed: {input_path}")
        logger.error(str(e))
        logger.debug(traceback.format_exc())
        return False


# =========================
# Batch Processing
# =========================
def batch_process(
    input_dir: str,
    output_dir: str,
    overwrite: bool = False
):
    """
    Batch N4 correction for MRI directory.
    """

    input_dir = Path(input_dir)
    output_dir = Path(output_dir)

    os.makedirs(output_dir, exist_ok=True)

    mri_files = sorted(list(input_dir.glob("*.nii.gz")))

    logger.info(f"Found {len(mri_files)} MRI files")
    logger.info(f"Input: {input_dir}")
    logger.info(f"Output: {output_dir}")

    success, fail, skip = 0, 0, 0

    for mri_file in tqdm(mri_files, desc="N4 Correction"):
        output_path = output_dir / mri_file.name

        if output_path.exists() and not overwrite:
            skip += 1
            continue

        ok = n4_bias_correction(
            input_path=str(mri_file),
            output_path=str(output_path)
        )

        if ok:
            success += 1
        else:
            fail += 1

    logger.info("=" * 50)
    logger.info(f"Done")
    logger.info(f"Success: {success}")
    logger.info(f"Fail: {fail}")
    logger.info(f"Skip: {skip}")
    logger.info("=" * 50)


# =========================
# CLI
# =========================
def main():
    parser = argparse.ArgumentParser(
        description="N4 Bias Field Correction for MRI"
    )

    # Single case
    parser.add_argument('--input', type=str, help='Input MRI path')
    parser.add_argument('--output', type=str, help='Output MRI path')
    parser.add_argument('--mask', type=str, help='Mask path (optional)')

    # Batch
    parser.add_argument('--input-dir', type=str, help='Input directory')
    parser.add_argument('--output-dir', type=str, help='Output directory')

    # Params
    parser.add_argument('--shrink', type=int, default=4)
    parser.add_argument('--overwrite', action='store_true')

    args = parser.parse_args()

    # ---- Single ----
    if args.input and args.output:
        n4_bias_correction(
            input_path=args.input,
            output_path=args.output,
            mask_path=args.mask,
            shrink_factor=args.shrink
        )
        return

    # ---- Batch ----
    if args.input_dir and args.output_dir:
        batch_process(
            input_dir=args.input_dir,
            output_dir=args.output_dir,
            overwrite=args.overwrite
        )
        return

    logger.error("Invalid arguments. Use single or batch mode.")


if __name__ == "__main__":
    main()
