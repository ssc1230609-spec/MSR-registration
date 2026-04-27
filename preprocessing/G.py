"""
G: Uniform Resizing

统一窗口尺寸的中心裁剪脚本
- 分析所有样本的有值区域，计算统一的偶数窗口尺寸
- 对每个样本进行中心裁剪，使有值区域位于窗口中心
- 保持标签值不变
"""

import os
import re
import csv
import argparse
import numpy as np
import nibabel as nib
from glob import glob
from typing import Tuple, Optional, Dict, List


def case_id_from_filename(path: str) -> str:
    """从文件名中提取病例ID"""
    base = os.path.basename(path)
    # 匹配 xxx_0000.nii.gz 或 xxx_0001.nii.gz (新格式)
    m = re.match(r"(.+?)_000[01]\.nii\.gz$", base)
    if m:
        return m.group(1)
    # 匹配 xxx_ct.nii.gz 或 xxx_mr.nii.gz (旧格式)
    m = re.match(r"(.+?)_(ct|mr)\.nii\.gz$", base, re.IGNORECASE)
    if m:
        return m.group(1)
    # 对于分割文件，直接去除.nii.gz后缀
    return base.replace(".nii.gz", "")


def build_file_map(directory: str, pattern: str = "*.nii.gz") -> Dict[str, str]:
    """构建文件映射字典"""
    files = glob(os.path.join(directory, pattern))
    file_map = {}
    for f in files:
        case_id = case_id_from_filename(f)
        file_map[case_id] = f
    return file_map


def compute_nonzero_bbox(data: np.ndarray, tol: float = 1e-6) -> Optional[Tuple[int, int, int, int, int, int]]:
    """计算非零区域的边界框"""
    mask = np.abs(data) > tol
    if not mask.any():
        return None
    coords = np.where(mask)
    x_min, x_max = int(coords[0].min()), int(coords[0].max())
    y_min, y_max = int(coords[1].min()), int(coords[1].max())
    z_min, z_max = int(coords[2].min()), int(coords[2].max())
    return (x_min, x_max, y_min, y_max, z_min, z_max)


def bbox_size(bbox: Tuple[int, int, int, int, int, int]) -> Tuple[int, int, int]:
    """计算边界框的尺寸"""
    x_min, x_max, y_min, y_max, z_min, z_max = bbox
    return (x_max - x_min + 1, y_max - y_min + 1, z_max - z_min + 1)


def union_bbox(b1: Optional[Tuple], b2: Optional[Tuple]) -> Optional[Tuple]:
    """计算两个边界框的并集"""
    if b1 is None:
        return b2
    if b2 is None:
        return b1
    return (
        min(b1[0], b2[0]), max(b1[1], b2[1]),
        min(b1[2], b2[2]), max(b1[3], b2[3]),
        min(b1[4], b2[4]), max(b1[5], b2[5]),
    )


def make_even(size: int) -> int:
    """将尺寸调整为偶数"""
    return size if size % 2 == 0 else size + 1


def pad_to_shape(data: np.ndarray, target_shape: Tuple[int, int, int], 
                 pad_value: float = 0.0) -> Tuple[np.ndarray, Tuple[int, int, int]]:
    """
    对数据进行均匀padding以达到目标形状
    返回：(填充后的数据, (x偏移, y偏移, z偏移))
    """
    current_shape = data.shape
    
    # 计算每个维度需要padding的量
    pad_x = max(0, target_shape[0] - current_shape[0])
    pad_y = max(0, target_shape[1] - current_shape[1])
    pad_z = max(0, target_shape[2] - current_shape[2])
    
    # 均匀padding（两边各一半）
    pad_x_before = pad_x // 2
    pad_x_after = pad_x - pad_x_before
    pad_y_before = pad_y // 2
    pad_y_after = pad_y - pad_y_before
    pad_z_before = pad_z // 2
    pad_z_after = pad_z - pad_z_before
    
    # 执行padding
    padded_data = np.pad(
        data,
        ((pad_x_before, pad_x_after),
         (pad_y_before, pad_y_after),
         (pad_z_before, pad_z_after)),
        mode='constant',
        constant_values=pad_value
    )
    
    # 返回偏移量（用于调整bbox和affine）
    offset = (pad_x_before, pad_y_before, pad_z_before)
    
    return padded_data, offset


def center_crop_bbox(data_shape: Tuple[int, int, int], 
                     content_bbox: Tuple[int, int, int, int, int, int],
                     target_size: Tuple[int, int, int]) -> Tuple[int, int, int, int, int, int]:
    """
    计算中心裁剪的边界框
    使有值区域位于目标窗口的中心
    """
    x_min, x_max, y_min, y_max, z_min, z_max = content_bbox
    target_x, target_y, target_z = target_size
    
    # 计算有值区域的中心
    content_center_x = (x_min + x_max) / 2.0
    content_center_y = (y_min + y_max) / 2.0
    content_center_z = (z_min + z_max) / 2.0
    
    # 计算裁剪窗口的起始位置（使有值区域中心对齐到窗口中心）
    crop_x_start = int(content_center_x - target_x / 2.0)
    crop_y_start = int(content_center_y - target_y / 2.0)
    crop_z_start = int(content_center_z - target_z / 2.0)
    
    # 确保不超出边界
    crop_x_start = max(0, min(crop_x_start, data_shape[0] - target_x))
    crop_y_start = max(0, min(crop_y_start, data_shape[1] - target_y))
    crop_z_start = max(0, min(crop_z_start, data_shape[2] - target_z))
    
    crop_x_end = crop_x_start + target_x - 1
    crop_y_end = crop_y_start + target_y - 1
    crop_z_end = crop_z_start + target_z - 1
    
    return (crop_x_start, crop_x_end, crop_y_start, crop_y_end, crop_z_start, crop_z_end)


def crop_data(data: np.ndarray, bbox: Tuple[int, int, int, int, int, int]) -> np.ndarray:
    """根据边界框裁剪数据"""
    x_min, x_max, y_min, y_max, z_min, z_max = bbox
    return data[x_min:x_max+1, y_min:y_max+1, z_min:z_max+1]


def adjust_affine(affine: np.ndarray, offset_xyz: Tuple[int, int, int]) -> np.ndarray:
    """调整仿射矩阵以反映裁剪偏移"""
    offset = np.array([offset_xyz[0], offset_xyz[1], offset_xyz[2]], dtype=float)
    new_affine = affine.copy()
    translation = affine[:3, :3].dot(offset)
    new_affine[:3, 3] = affine[:3, 3] + translation
    return new_affine


def save_nifti(data: np.ndarray, like_img: nib.Nifti1Image, 
               offset_xyz: Tuple[int, int, int], out_path: str):
    """保存NIfTI文件，保持原始数据类型和元数据"""
    new_aff = adjust_affine(like_img.affine, offset_xyz)
    hdr = like_img.header.copy()
    data_dtype = like_img.get_data_dtype()
    img = nib.Nifti1Image(data.astype(data_dtype), new_aff, header=hdr)
    img.set_qform(new_aff)
    img.set_sform(new_aff)
    nib.save(img, out_path)


def analyze_all_samples(vol_dir: str, seg_dir: str, tol: float = 1e-6) -> Tuple[List[Dict], Tuple[int, int, int]]:
    """
    分析所有样本的有值区域
    返回：(样本信息列表, 统一的目标尺寸)
    """
    print("=== 开始分析所有样本的有值区域 ===")
    
    # 构建文件映射
    ct_map = build_file_map(vol_dir, "*_ct.nii.gz")
    mr_map = build_file_map(vol_dir, "*_mr.nii.gz")
    seg_map = build_file_map(seg_dir, "*.nii.gz")
    
    # 获取所有病例ID
    all_case_ids = sorted(set(ct_map.keys()) | set(mr_map.keys()))
    
    samples_info = []
    max_sizes = [0, 0, 0]  # [x, y, z]
    
    for case_id in all_case_ids:
        ct_path = ct_map.get(case_id)
        mr_path = mr_map.get(case_id)
        seg_path = seg_map.get(case_id)
        
        if not ct_path and not mr_path:
            print(f"[跳过] {case_id}: 未找到CT或MR文件")
            continue
        
        # 加载数据并计算边界框
        union_bbox_val = None
        shape = None
        
        if ct_path:
            ct_img = nib.load(ct_path)
            ct_data = ct_img.get_fdata()
            shape = ct_data.shape
            ct_bbox = compute_nonzero_bbox(ct_data, tol)
            union_bbox_val = union_bbox(union_bbox_val, ct_bbox)
        
        if mr_path:
            mr_img = nib.load(mr_path)
            mr_data = mr_img.get_fdata()
            if shape is None:
                shape = mr_data.shape
            if mr_data.shape == shape:
                mr_bbox = compute_nonzero_bbox(mr_data, tol)
                union_bbox_val = union_bbox(union_bbox_val, mr_bbox)
            else:
                print(f"[警告] {case_id}: MR形状与CT不一致")
        
        if union_bbox_val is None:
            print(f"[跳过] {case_id}: 有值区域为空")
            continue
        
        # 计算有值区域尺寸
        size = bbox_size(union_bbox_val)
        
        # 更新最大尺寸
        max_sizes[0] = max(max_sizes[0], size[0])
        max_sizes[1] = max(max_sizes[1], size[1])
        max_sizes[2] = max(max_sizes[2], size[2])
        
        samples_info.append({
            'case_id': case_id,
            'ct_path': ct_path,
            'mr_path': mr_path,
            'seg_path': seg_path,
            'shape': shape,
            'content_bbox': union_bbox_val,
            'content_size': size
        })
        
        print(f"[分析] {case_id}: 形状={shape}, 有值区域尺寸={size}")
    
    # 将最大尺寸调整为偶数
    target_size = tuple(make_even(s) for s in max_sizes)
    
    print(f"\n样本数量: {len(samples_info)}")
    print(f"最大有值区域尺寸: {max_sizes}")
    print(f"统一目标窗口尺寸（偶数）: {target_size}")
    print("=" * 50)
    
    return samples_info, target_size


def crop_all_samples(samples_info: List[Dict], target_size: Tuple[int, int, int],
                     out_vol_dir: str, out_seg_dir: str):
    """对所有样本进行统一窗口的中心裁剪（必要时先padding）"""
    os.makedirs(out_vol_dir, exist_ok=True)
    os.makedirs(out_seg_dir, exist_ok=True)
    
    print(f"\n=== 开始处理，目标窗口尺寸: {target_size} ===\n")
    
    crop_records = []
    
    for idx, info in enumerate(samples_info, 1):
        case_id = info['case_id']
        ct_path = info['ct_path']
        mr_path = info['mr_path']
        seg_path = info['seg_path']
        shape = info['shape']
        content_bbox = info['content_bbox']
        
        # 检查是否需要padding
        needs_padding = (target_size[0] > shape[0] or target_size[1] > shape[1] or target_size[2] > shape[2])
        if needs_padding:
            print(f"[{idx}/{len(samples_info)}] 处理 {case_id} (需要padding)")
        else:
            print(f"[{idx}/{len(samples_info)}] 处理 {case_id}")
        
        print(f"  原始形状: {shape}")
        print(f"  有值区域: [{content_bbox[0]}:{content_bbox[1]}, {content_bbox[2]}:{content_bbox[3]}, {content_bbox[4]}:{content_bbox[5]}]")
        
        # 加载所有数据
        ct_img = None
        ct_data = None
        mr_img = None
        mr_data = None
        seg_img = None
        seg_data = None
        
        if ct_path:
            ct_img = nib.load(ct_path)
            ct_data = ct_img.get_fdata()
        
        if mr_path:
            mr_img = nib.load(mr_path)
            mr_data = mr_img.get_fdata()
            if mr_data.shape != shape:
                print(f"  [警告] MR形状与CT不一致，跳过MR")
                mr_img = None
                mr_data = None
        
        if seg_path:
            seg_img = nib.load(seg_path)
            seg_data = seg_img.get_fdata()
            if seg_data.shape != shape:
                print(f"  [警告] 标签形状与CT不一致，跳过标签")
                seg_img = None
                seg_data = None
        
        # 如果需要padding，先进行padding
        pad_offset = (0, 0, 0)
        adjusted_bbox = content_bbox
        working_shape = shape
        
        if needs_padding:
            print(f"  执行padding至 {target_size}")
            
            if ct_data is not None:
                ct_data, pad_offset = pad_to_shape(ct_data, target_size, pad_value=0.0)
            
            if mr_data is not None:
                mr_data, _ = pad_to_shape(mr_data, target_size, pad_value=0.0)
            
            if seg_data is not None:
                seg_data, _ = pad_to_shape(seg_data, target_size, pad_value=0.0)
            
            # 调整bbox以反映padding
            adjusted_bbox = (
                content_bbox[0] + pad_offset[0],
                content_bbox[1] + pad_offset[0],
                content_bbox[2] + pad_offset[1],
                content_bbox[3] + pad_offset[1],
                content_bbox[4] + pad_offset[2],
                content_bbox[5] + pad_offset[2]
            )
            working_shape = target_size
        
        # 计算中心裁剪边界框
        crop_bbox = center_crop_bbox(working_shape, adjusted_bbox, target_size)
        x_min, x_max, y_min, y_max, z_min, z_max = crop_bbox
        
        print(f"  裁剪窗口: [{x_min}:{x_max}, {y_min}:{y_max}, {z_min}:{z_max}]")
        
        # 裁剪并保存CT
        if ct_data is not None:
            ct_cropped = crop_data(ct_data, crop_bbox)
            out_ct = os.path.join(out_vol_dir, os.path.basename(ct_path))
            # affine偏移需要考虑padding的负偏移
            affine_offset = (x_min - pad_offset[0], y_min - pad_offset[1], z_min - pad_offset[2])
            save_nifti(ct_cropped, ct_img, affine_offset, out_ct)
            print(f"  已保存CT: {os.path.basename(out_ct)}, 形状: {ct_cropped.shape}")
        
        # 裁剪并保存MR
        if mr_data is not None:
            mr_cropped = crop_data(mr_data, crop_bbox)
            out_mr = os.path.join(out_vol_dir, os.path.basename(mr_path))
            affine_offset = (x_min - pad_offset[0], y_min - pad_offset[1], z_min - pad_offset[2])
            save_nifti(mr_cropped, mr_img, affine_offset, out_mr)
            print(f"  已保存MR: {os.path.basename(out_mr)}, 形状: {mr_cropped.shape}")
        
        # 裁剪并保存分割标签
        if seg_data is not None:
            original_labels = np.unique(seg_data)
            seg_cropped = crop_data(seg_data, crop_bbox)
            out_seg = os.path.join(out_seg_dir, os.path.basename(seg_path))
            affine_offset = (x_min - pad_offset[0], y_min - pad_offset[1], z_min - pad_offset[2])
            save_nifti(seg_cropped, seg_img, affine_offset, out_seg)
            
            # 验证标签值未改变
            cropped_labels = np.unique(seg_cropped)
            if not np.array_equal(np.sort(original_labels), np.sort(cropped_labels)):
                print(f"  [警告] 标签值发生变化！原始: {original_labels}, 裁剪后: {cropped_labels}")
            else:
                print(f"  已保存标签: {os.path.basename(out_seg)}, 形状: {seg_cropped.shape}, 标签值未变")
        
        crop_records.append([
            case_id, 
            shape[0], shape[1], shape[2],
            content_bbox[0], content_bbox[1], content_bbox[2], 
            content_bbox[3], content_bbox[4], content_bbox[5],
            pad_offset[0], pad_offset[1], pad_offset[2],
            x_min, x_max, y_min, y_max, z_min, z_max,
            target_size[0], target_size[1], target_size[2]
        ])
        print()
    
    # 保存裁剪记录
    csv_path = os.path.join(out_vol_dir, "crop_records.csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "case_id", 
            "orig_x", "orig_y", "orig_z",
            "content_x_min", "content_x_max", "content_y_min", 
            "content_y_max", "content_z_min", "content_z_max",
            "pad_x", "pad_y", "pad_z",
            "crop_x_min", "crop_x_max", "crop_y_min", 
            "crop_y_max", "crop_z_min", "crop_z_max",
            "final_x", "final_y", "final_z"
        ])
        writer.writerows(crop_records)
    
    print(f"裁剪记录已保存至: {csv_path}")
    print("=== 裁剪完成 ===")


def main():
    parser = argparse.ArgumentParser(
        description="统一窗口尺寸的中心裁剪，使有值区域位于窗口中心"
    )
    parser.add_argument("--vol_dir", type=str, required=True,
                        help="影像数据目录（包含CT和MR）")
    parser.add_argument("--seg_dir", type=str, required=True,
                        help="分割标签目录")
    parser.add_argument("--out_vol_dir", type=str, required=True,
                        help="裁剪后影像输出目录")
    parser.add_argument("--out_seg_dir", type=str, required=True,
                        help="裁剪后标签输出目录")
    parser.add_argument("--tol", type=float, default=1e-6,
                        help="判断非零值的阈值")
    
    args = parser.parse_args()
    
    # 第一步：分析所有样本，确定统一的目标尺寸
    samples_info, target_size = analyze_all_samples(
        vol_dir=args.vol_dir,
        seg_dir=args.seg_dir,
        tol=args.tol
    )
    
    if not samples_info:
        print("未找到有效样本，退出")
        return
    
    # 第二步：对所有样本进行统一窗口的中心裁剪
    crop_all_samples(
        samples_info=samples_info,
        target_size=target_size,
        out_vol_dir=args.out_vol_dir,
        out_seg_dir=args.out_seg_dir
    )


if __name__ == "__main__":
    main()
