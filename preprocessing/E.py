"""
E：Pre-Registration
"""
#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
MRI-to-CT Rigid Registration using ANTs
Features:
- Mutual Information (MI)-based rigid registration
- Intensity normalization
- Single-case and batch processing
- Robust ID matching
- Logging support
Author: zbh
"""

import ants
import os
import re
import argparse
import logging
import traceback
from typing import Optional, List


# =========================
# Logging
# =========================
logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] %(levelname)s: %(message)s'
)
logger = logging.getLogger(__name__)


# =========================
# Utils
# =========================
def extract_id(filename: str) -> Optional[str]:
    """
    Extract subject ID from filename.

    Supported patterns:
    - Spine_XXX_
    - cXXX_ or cXXX-XXX_

    Args:
        filename: file name string

    Returns:
        ID string or None
    """
    match = re.search(r'Spine_(\d+)_', filename)
    if match:
        return match.group(1)

    match = re.search(r'c([\d-]+)_', filename)
    if match:
        return match.group(1)

    return None


# =========================
# Core Registration
# =========================
def rigid_register_mri_to_ct(
    ct_path: str,
    mri_path: str,
    output_path: str,
    normalize: bool = True
) -> float:
    """
    Perform rigid registration of MRI to CT.

    Args:
        ct_path: CT image (fixed)
        mri_path: MRI image (moving)
        output_path: output file path
        normalize: whether to normalize intensity

    Returns:
        mutual information value
    """

    logger.info(f"Registering MRI -> CT")
    logger.info(f"CT: {ct_path}")
    logger.info(f"MRI: {mri_path}")

    # Load images
    ct_img = ants.image_read(ct_path)
    mri_img = ants.image_read(mri_path)

    # Normalize (important for MI)
    if normalize:
        ct_norm = ants.iMath_normalize(ct_img)
        mri_norm = ants.iMath_normalize(mri_img)
    else:
        ct_norm = ct_img
        mri_norm = mri_img

    # Rigid registration
    reg = ants.registration(
        fixed=ct_norm,
        moving=mri_norm,
        type_of_transform='Rigid',
        aff_metric='MI',
        aff_sampling=32,
        aff_iterations=(160, 80, 40),
        aff_smoothing_sigmas=(2, 1, 0),
        aff_shrink_factors=(2, 1, 1)
    )

    # Compute MI after registration
    warped_norm = ants.apply_transforms(
        fixed=ct_norm,
        moving=mri_norm,
        transformlist=reg['fwdtransforms']
    )

    mi = ants.image_mutual_information(ct_norm, warped_norm)

    # Apply transform to original image
    warped = ants.apply_transforms(
        fixed=ct_img,
        moving=mri_img,
        transformlist=reg['fwdtransforms'],
        interpolator="bSpline"
    )

    # Save result
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    ants.image_write(warped, output_path)

    logger.info(f"Done. MI = {mi:.4f}")
    logger.info(f"Saved to: {output_path}")

    return float(mi)


# =========================
# Batch Processing
# =========================
def batch_registration(
    ct_dir: str,
    mri_dir: str,
    output_dir: str
):
    """
    Batch MRI-to-CT registration.
    """
    ct_files = [f for f in os.listdir(ct_dir) if f.endswith('.nii.gz')]
    mri_files = [f for f in os.listdir(mri_dir) if f.endswith('.nii.gz')]

    os.makedirs(output_dir, exist_ok=True)

    logger.info(f"CT files: {len(ct_files)}")
    logger.info(f"MRI files: {len(mri_files)}")

    for ct_file in ct_files:
        ct_id = extract_id(ct_file)

        if ct_id is None:
            logger.warning(f"Cannot extract ID from CT: {ct_file}")
            continue

        matched_mris = [m for m in mri_files if extract_id(m) == ct_id]

        if not matched_mris:
            logger.warning(f"No MRI found for CT {ct_file} (ID={ct_id})")
            continue

        for mri_file in matched_mris:
            ct_path = os.path.join(ct_dir, ct_file)
            mri_path = os.path.join(mri_dir, mri_file)
            output_path = os.path.join(output_dir, mri_file)

            try:
                rigid_register_mri_to_ct(ct_path, mri_path, output_path)
            except Exception as e:
                logger.error(f"Error processing {ct_file} & {mri_file}")
                logger.error(str(e))
                logger.debug(traceback.format_exc())


# =========================
# CLI
# =========================
def main():
    parser = argparse.ArgumentParser(
        description="MRI-to-CT Rigid Registration (ANTs)"
    )

    # Single case
    parser.add_argument('--ct', type=str, help='CT path (fixed)')
    parser.add_argument('--mri', type=str, help='MRI path (moving)')
    parser.add_argument('--output', type=str, help='Output path')

    # Batch
    parser.add_argument('--ct-dir', type=str, help='CT directory')
    parser.add_argument('--mri-dir', type=str, help='MRI directory')
    parser.add_argument('--out-dir', type=str, help='Output directory')

    args = parser.parse_args()

    # ---- Single case ----
    if args.ct and args.mri and args.output:
        try:
            rigid_register_mri_to_ct(args.ct, args.mri, args.output)
        except Exception as e:
            logger.error("Single-case registration failed")
            logger.error(str(e))
            logger.debug(traceback.format_exc())
        return

    # ---- Batch ----
    if args.ct_dir and args.mri_dir and args.out_dir:
        batch_registration(args.ct_dir, args.mri_dir, args.out_dir)
        return

    # ---- Invalid ----
    logger.error("Invalid arguments. Use either single or batch mode.")


if __name__ == "__main__":
    main()
