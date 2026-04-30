"""
F：Mask-guided adaptive cropping
Author: zbh
"""
#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import nibabel as nib
import numpy as np
from pathlib import Path


# -----------------------------
# 1. Bounding box extraction
# -----------------------------
def get_label_bbox(label_data, margin=5):
    """
    Compute bounding box from label volume.

    Args:
        label_data (np.ndarray): 3D label volume
        margin (int): margin for expansion

    Returns:
        tuple or None: (x_min, x_max, y_min, y_max, z_min, z_max)
    """
    coords = np.where(label_data > 0)

    if len(coords[0]) == 0:
        print("[Warning] Empty label volume.")
        return None

    x_min, x_max = coords[0].min(), coords[0].max()
    y_min, y_max = coords[1].min(), coords[1].max()
    z_min, z_max = coords[2].min(), coords[2].max()

    shape = label_data.shape

    x_min = max(0, x_min - margin)
    y_min = max(0, y_min - margin)
    z_min = max(0, z_min - margin)

    x_max = min(shape[0] - 1, x_max + margin)
    y_max = min(shape[1] - 1, y_max + margin)
    z_max = min(shape[2] - 1, z_max + margin)

    return int(x_min), int(x_max), int(y_min), int(y_max), int(z_min), int(z_max)


# -----------------------------
# 2. NIfTI cropping function
# -----------------------------
def crop_nifti_by_bbox(img_path, bbox, output_path, is_label=False):
    """
    Crop a NIfTI image using bounding box.

    Args:
        img_path (str): input path
        bbox (tuple): bounding box
        output_path (str): output path
        is_label (bool): whether input is label

    Returns:
        bool
    """
    try:
        img = nib.load(img_path)

        data = img.get_fdata()

        if is_label:
            dtype = img.header.get_data_dtype()
            if np.issubdtype(dtype, np.integer):
                data = data.astype(dtype)
            else:
                data = data.astype(np.int16)

        x_min, x_max, y_min, y_max, z_min, z_max = bbox

        cropped = data[
            x_min:x_max + 1,
            y_min:y_max + 1,
            z_min:z_max + 1
        ]

        # update affine
        affine = img.affine.copy()
        offset = np.array([x_min, y_min, z_min])
        affine[:3, 3] = affine[:3, 3] + affine[:3, :3] @ offset

        new_img = nib.Nifti1Image(cropped, affine)

        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        nib.save(new_img, output_path)

        return True

    except Exception as e:
        print(f"[Error] Crop failed: {img_path}, {e}")
        return False


# -----------------------------
# 3. Patient processing
# -----------------------------
def process_patient(ct_path, mri_path, label_path, out_dir, pid, margin=5):
    """
    Process one patient (CT + MRI + Label)
    """
    print(f"\n[Processing] {pid}")

    label = nib.load(label_path).get_fdata()
    bbox = get_label_bbox(label, margin)

    if bbox is None:
        print(f"[Skip] No valid label region: {pid}")
        return False

    print(f"  BBox: {bbox}")

    ct_out = os.path.join(out_dir, "CT", os.path.basename(ct_path))
    mri_out = os.path.join(out_dir, "MRI", os.path.basename(mri_path))
    lab_out = os.path.join(out_dir, "Label", os.path.basename(label_path))

    ok1 = crop_nifti_by_bbox(ct_path, bbox, ct_out, is_label=False)
    ok2 = crop_nifti_by_bbox(mri_path, bbox, mri_out, is_label=False)
    ok3 = crop_nifti_by_bbox(label_path, bbox, lab_out, is_label=True)

    if ok1 and ok2 and ok3:
        print(f"[Done] {pid}")
        return True
    else:
        print(f"[Failed] {pid}")
        return False


# -----------------------------
# 4. File matching
# -----------------------------
def find_pairs(volume_dir, label_dir):
    """
    Match CT/MRI/Label by patient id.
    """
    vol_files = {f.stem.replace(".nii", ""): f for f in Path(volume_dir).glob("*.nii.gz")}
    lab_files = {f.stem.replace(".nii", ""): f for f in Path(label_dir).glob("*.nii.gz")}

    pairs = []

    for lid, lpath in lab_files.items():
        ct_key = f"{lid}_ct"
        mri_key = f"{lid}_mr"

        if ct_key in vol_files and mri_key in vol_files:
            pairs.append((
                str(vol_files[ct_key]),
                str(vol_files[mri_key]),
                str(lpath),
                lid
            ))
        else:
            print(f"[Warning] Missing: {lid}")

    return pairs


# -----------------------------
# 5. Main
# -----------------------------
def main():
    volume_dir = "./data/Public/volumes_73"
    label_dir = "./data/Public/HN_seg_73"
    output_dir = "./data/Public/cropped_by_label"

    margin = 10

    print("=" * 60)
    print("Label-guided Cropping Tool")
    print("=" * 60)

    pairs = find_pairs(volume_dir, label_dir)

    print(f"Found {len(pairs)} cases\n")

    success = 0

    for ct, mri, lab, pid in pairs:
        if process_patient(ct, mri, lab, output_dir, pid, margin):
            success += 1

    print("\n" + "=" * 60)
    print(f"Finished: {success}/{len(pairs)} successful")
    print("=" * 60)


if __name__ == "__main__":
    main()
