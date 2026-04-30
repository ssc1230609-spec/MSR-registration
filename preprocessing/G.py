"""
G: Uniform Resizing
Author: zbh
Uniformly sized key plot elements

- Analyze the valuable regions of all samples and calculate a uniform even-numbered window size

- Center each sample to ensure that regions centered within the window are valuable

- Keep label values ​​unchanged
"""
#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import re
import csv
import argparse
from glob import glob
from typing import Tuple, Optional, Dict, List

import numpy as np
import nibabel as nib


# -----------------------------
# 1. File utilities
# -----------------------------
def case_id_from_filename(path: str) -> str:
    """Extract case ID from filename."""
    base = os.path.basename(path)

    m = re.match(r"(.+?)_000[01]\.nii\.gz$", base)
    if m:
        return m.group(1)

    m = re.match(r"(.+?)_(ct|mr)\.nii\.gz$", base, re.IGNORECASE)
    if m:
        return m.group(1)

    return base.replace(".nii.gz", "")


def build_file_map(directory: str, pattern="*.nii.gz") -> Dict[str, str]:
    """Build case-id → file-path mapping."""
    files = glob(os.path.join(directory, pattern))
    return {case_id_from_filename(f): f for f in files}


# -----------------------------
# 2. Geometry utilities
# -----------------------------
def compute_bbox(data: np.ndarray, tol=1e-6):
    """Compute foreground bounding box."""
    mask = np.abs(data) > tol
    if not mask.any():
        return None

    x, y, z = np.where(mask)
    return int(x.min()), int(x.max()), int(y.min()), int(y.max()), int(z.min()), int(z.max())


def bbox_size(bbox):
    x0, x1, y0, y1, z0, z1 = bbox
    return (x1 - x0 + 1, y1 - y0 + 1, z1 - z0 + 1)


def union_bbox(b1, b2):
    if b1 is None:
        return b2
    if b2 is None:
        return b1

    return (
        min(b1[0], b2[0]), max(b1[1], b2[1]),
        min(b1[2], b2[2]), max(b1[3], b2[3]),
        min(b1[4], b2[4]), max(b1[5], b2[5]),
    )


def make_even(x: int) -> int:
    return x if x % 2 == 0 else x + 1


# -----------------------------
# 3. Cropping & padding
# -----------------------------
def pad_to_shape(data, target_shape, value=0.0):
    """Symmetric padding."""
    pad = [max(0, t - s) for t, s in zip(target_shape, data.shape)]
    before = [p // 2 for p in pad]
    after = [p - b for p, b in zip(pad, before)]

    padded = np.pad(
        data,
        tuple((b, a) for b, a in zip(before, after)),
        mode="constant",
        constant_values=value,
    )
    return padded, (before[0], before[1], before[2])


def crop(data, bbox):
    x0, x1, y0, y1, z0, z1 = bbox
    return data[x0:x1 + 1, y0:y1 + 1, z0:z1 + 1]


def center_crop_bbox(shape, content_bbox, target):
    """Center foreground in fixed-size window."""
    x0, x1, y0, y1, z0, z1 = content_bbox

    cx = (x0 + x1) / 2
    cy = (y0 + y1) / 2
    cz = (z0 + z1) / 2

    sx, sy, sz = target

    start_x = int(cx - sx / 2)
    start_y = int(cy - sy / 2)
    start_z = int(cz - sz / 2)

    start_x = max(0, min(start_x, shape[0] - sx))
    start_y = max(0, min(start_y, shape[1] - sy))
    start_z = max(0, min(start_z, shape[2] - sz))

    return (
        start_x, start_x + sx - 1,
        start_y, start_y + sy - 1,
        start_z, start_z + sz - 1,
    )


# -----------------------------
# 4. I/O
# -----------------------------
def save_nifti(data, ref_img, offset, path):
    """Save NIfTI with corrected affine."""
    affine = ref_img.affine.copy()
    shift = affine[:3, :3] @ np.array(offset)
    affine[:3, 3] += shift

    img = nib.Nifti1Image(data.astype(ref_img.get_data_dtype()), affine, header=ref_img.header)
    nib.save(img, path)


# -----------------------------
# 5. Global analysis
# -----------------------------
def analyze(vol_dir, seg_dir, tol):
    print("\n[1] Analyzing dataset...")

    ct_map = build_file_map(vol_dir, "*_ct.nii.gz")
    mr_map = build_file_map(vol_dir, "*_mr.nii.gz")
    seg_map = build_file_map(seg_dir)

    cases = sorted(set(ct_map) | set(mr_map))

    infos = []
    max_size = [0, 0, 0]

    for cid in cases:
        bbox = None
        shape = None

        for mp in [ct_map, mr_map]:
            if cid not in mp:
                continue
            img = nib.load(mp[cid])
            data = img.get_fdata()
            shape = data.shape
            bbox = union_bbox(bbox, compute_bbox(data, tol))

        if bbox is None:
            continue

        size = bbox_size(bbox)
        max_size = [max(a, b) for a, b in zip(max_size, size)]

        infos.append({"id": cid, "bbox": bbox, "shape": shape})

        print(f"{cid}: shape={shape}, fg={size}")

    target = tuple(make_even(x) for x in max_size)

    print(f"\nTarget window: {target}\n")
    return infos, target


# -----------------------------
# 6. Processing
# -----------------------------
def process(infos, target, vol_dir, seg_dir, out_v, out_s):
    os.makedirs(out_v, exist_ok=True)
    os.makedirs(out_s, exist_ok=True)

    records = []

    for i, info in enumerate(infos):
        cid = info["id"]

        ct = build_file_map(vol_dir).get(cid + "_ct")
        mr = build_file_map(vol_dir).get(cid + "_mr")
        seg = build_file_map(seg_dir).get(cid)

        if not ct and not mr:
            continue

        print(f"[{i+1}/{len(infos)}] {cid}")

        ref_img = nib.load(ct or mr)
        shape = ref_img.shape

        bbox = center_crop_bbox(shape, info["bbox"], target)

        def run(path, outdir, is_seg=False):
            if not path:
                return
            img = nib.load(path)
            data = img.get_fdata()
            cropped = crop(data, bbox)
            out = os.path.join(outdir, os.path.basename(path))
            save_nifti(cropped, img, (0, 0, 0), out)

        run(ct, out_v)
        run(mr, out_v)
        run(seg, out_s, True)

        records.append([cid, shape, bbox])

    with open(os.path.join(out_v, "records.csv"), "w") as f:
        writer = csv.writer(f)
        writer.writerow(["id", "shape", "bbox"])
        writer.writerows(records)


# -----------------------------
# 7. Main
# -----------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--vol_dir", required=True)
    parser.add_argument("--seg_dir", required=True)
    parser.add_argument("--out_vol", required=True)
    parser.add_argument("--out_seg", required=True)
    parser.add_argument("--tol", type=float, default=1e-6)
    args = parser.parse_args()

    infos, target = analyze(args.vol_dir, args.seg_dir, args.tol)
    process(infos, target, args.vol_dir, args.seg_dir, args.out_vol, args.out_seg)


if __name__ == "__main__":
    main()
