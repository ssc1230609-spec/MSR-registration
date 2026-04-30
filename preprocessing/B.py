"""
B: Preliminary cropping
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
NIfTI Cropping Tool

Features:
- Crop along arbitrary axis (X/Y/Z)
- Batch and single-case processing
- Boundary-safe cropping
- Logging + CLI support

Author: Your Name
"""

import os
import argparse
import logging
import traceback
from pathlib import Path
from typing import Tuple

import nibabel as nib
import numpy as np
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
def crop_nifti(
    input_path: str,
    output_path: str,
    axis: int = 0,
    start: int = 0,
    end: int = None
) -> bool:
    """
    Crop NIfTI image along a given axis.

    Args:
        input_path: input .nii.gz file
        output_path: output file
        axis: axis to crop (0=X, 1=Y, 2=Z)
        start: start index
        end: end index

    Returns:
        success flag
    """

    try:
        logger.info(f"Processing: {input_path}")

        img = nib.load(input_path)
        data = img.get_fdata()

        shape = data.shape
        logger.info(f"Original shape: {shape}")

        # Boundary check
        if end is None or end > shape[axis]:
            end = shape[axis]

        if start < 0 or start >= end:
            raise ValueError(f"Invalid crop range: start={start}, end={end}")

        # Build slicing
        slices = [slice(None)] * 3
        slices[axis] = slice(start, end)

        cropped_data = data[tuple(slices)]

        logger.info(f"Cropped shape: {cropped_data.shape}")

        # Adjust affine (critical!)
        affine = img.affine.copy()
        voxel_size = affine[:3, :3]

        # 更新 origin（关键，否则空间错位）
        offset = np.zeros(3)
        offset[axis] = start
        translation = voxel_size @ offset

        affine[:3, 3] += translation

        # Create new image
        new_img = nib.Nifti1Image(cropped_data, affine, img.header)

        # Save
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        nib.save(new_img, output_path)

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
def batch_crop(
    input_dir: str,
    output_dir: str,
    axis: int,
    start: int,
    end: int,
    overwrite: bool = False
):
    """
    Batch crop NIfTI files in a directory.
    """

    input_dir = Path(input_dir)
    output_dir = Path(output_dir)

    os.makedirs(output_dir, exist_ok=True)

    files = sorted(list(input_dir.glob("*.nii.gz")))

    logger.info(f"Found {len(files)} files")
    logger.info(f"Axis: {axis}, Range: [{start}, {end}]")

    success, fail, skip = 0, 0, 0

    for f in tqdm(files, desc="Cropping"):
        out_path = output_dir / f.name

        if out_path.exists() and not overwrite:
            skip += 1
            continue

        ok = crop_nifti(
            input_path=str(f),
            output_path=str(out_path),
            axis=axis,
            start=start,
            end=end
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
        description="NIfTI Cropping Tool"
    )

    # Single
    parser.add_argument('--input', type=str, help='Input file')
    parser.add_argument('--output', type=str, help='Output file')

    # Batch
    parser.add_argument('--input-dir', type=str, help='Input directory')
    parser.add_argument('--output-dir', type=str, help='Output directory')

    # Params
    parser.add_argument('--axis', type=int, default=0, help='0=X,1=Y,2=Z')
    parser.add_argument('--start', type=int, default=0)
    parser.add_argument('--end', type=int, default=None)
    parser.add_argument('--overwrite', action='store_true')

    args = parser.parse_args()

    # ---- Single ----
    if args.input and args.output:
        crop_nifti(
            input_path=args.input,
            output_path=args.output,
            axis=args.axis,
            start=args.start,
            end=args.end
        )
        return

    # ---- Batch ----
    if args.input_dir and args.output_dir:
        batch_crop(
            input_dir=args.input_dir,
            output_dir=args.output_dir,
            axis=args.axis,
            start=args.start,
            end=args.end,
            overwrite=args.overwrite
        )
        return

    logger.error("Invalid arguments. Use single or batch mode.")


if __name__ == "__main__":
    main()
