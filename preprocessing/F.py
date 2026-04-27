"""
F：Mask-guided adaptive cropping
"""
import nibabel as nib
import numpy as np
import os
from pathlib import Path

def get_label_bbox(label_data, margin=5):
    """
    根据标签数据获取边界框
    
    Args:
        label_data: 标签数据
        margin: 边界扩展像素数
    
    Returns:
        tuple: (x_min, x_max, y_min, y_max, z_min, z_max)
    """
    # 找到所有非零区域
    nonzero_coords = np.where(label_data > 0)
    
    if len(nonzero_coords[0]) == 0:
        print("  警告: 标签中没有找到非零区域!")
        return None
    
    # 计算边界框
    x_min, x_max = int(nonzero_coords[0].min()), int(nonzero_coords[0].max())
    y_min, y_max = int(nonzero_coords[1].min()), int(nonzero_coords[1].max())
    z_min, z_max = int(nonzero_coords[2].min()), int(nonzero_coords[2].max())
    
    # 添加边界扩展
    shape = label_data.shape
    z_margin = margin  # z轴额外扩展10个像素
    x_margin = margin # x轴额外扩展10个像素
    y_margin = margin # y轴额外扩展10个像素
    
    x_min = max(0, x_min - x_margin)
    x_max = min(shape[0] - 1, x_max + x_margin)
    y_min = max(0, y_min - y_margin)
    y_max = min(shape[1] - 1, y_max + y_margin)
    z_min = max(0, z_min - z_margin)
    z_max = min(shape[2] - 1, z_max + z_margin)
    
    return (int(x_min), int(x_max), int(y_min), int(y_max), int(z_min), int(z_max))

def crop_nifti_by_bbox(img_path, bbox, output_path, is_label=False):
    """
    根据边界框裁剪NIfTI文件
    
    Args:
        img_path: 输入图像路径
        bbox: 边界框 (x_min, x_max, y_min, y_max, z_min, z_max)
        output_path: 输出路径
        is_label: 是否为标签文件
    
    Returns:
        bool: 是否成功
    """
    try:
        # 加载图像
        img = nib.load(img_path)
        
        # 根据是否为标签选择数据获取方式
        if is_label:
            # 标签文件：先获取原始数据类型，然后转换为合适的类型
            original_dtype = img.header.get_data_dtype()
            data = img.get_fdata()  # 先获取为浮点数
            
            # 转换回原始整数类型，但确保兼容性
            if np.issubdtype(original_dtype, np.integer):
                # 如果原始是整数类型，转换回去
                if original_dtype == np.uint8:
                    data = data.astype(np.uint8)
                elif original_dtype == np.int16:
                    data = data.astype(np.int16)
                elif original_dtype == np.uint16:
                    data = data.astype(np.uint16)
                else:
                    data = data.astype(np.int16)  # 默认使用int16
            else:
                data = data.astype(np.int16)
        else:
            # 图像文件使用浮点数
            data = img.get_fdata()
        
        affine = img.affine
        header = img.header
        
        # 解包边界框
        x_min, x_max, y_min, y_max, z_min, z_max = bbox
        
        # 裁剪数据
        cropped_data = data[x_min:x_max+1, y_min:y_max+1, z_min:z_max+1]
        
        # 更新仿射矩阵 - 调整原点位置
        new_affine = affine.copy()
        # 计算新的原点偏移
        offset = np.array([x_min, y_min, z_min])
        new_affine[:3, 3] = affine[:3, 3] + affine[:3, :3] @ offset
        
        # 更新header
        new_header = header.copy()
        new_header.set_data_shape(cropped_data.shape)
        
        # 如果是标签文件，确保header的数据类型正确
        if is_label:
            new_header.set_data_dtype(cropped_data.dtype)
        
        # 创建新的NIfTI图像
        new_img = nib.Nifti1Image(cropped_data, new_affine, new_header)
        
        # 保存文件
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        nib.save(new_img, output_path)
        
        return True
        
    except Exception as e:
        print(f"  裁剪失败: {e}")
        return False

def process_patient_set(ct_path, mri_path, label_path, output_base, patient_id, margin=5):
    """
    处理一套患者数据（CT、MRI、Label）
    
    Args:
        ct_path: CT文件路径
        mri_path: MRI文件路径
        label_path: Label文件路径
        output_base: 输出基础路径
        patient_id: 患者ID
        margin: 边界扩展像素数
    
    Returns:
        bool: 是否成功
    """
    try:
        print(f"处理患者: {patient_id}")
        
        # 加载label获取边界框
        label_img = nib.load(label_path)
        label_data = label_img.get_fdata()
        
        print(f"  Label形状: {label_data.shape}")
        print(f"  Label值范围: {label_data.min()} - {label_data.max()}")
        print(f"  非零体素数: {np.count_nonzero(label_data)}")
        
        # 获取边界框
        bbox = get_label_bbox(label_data, margin)
        
        if bbox is None:
            print(f"  跳过患者 {patient_id}: 无法获取有效边界框")
            return False
        
        x_min, x_max, y_min, y_max, z_min, z_max = bbox
        crop_size = (x_max - x_min + 1, y_max - y_min + 1, z_max - z_min + 1)
        
        print(f"  边界框: x[{x_min}:{x_max}] y[{y_min}:{y_max}] z[{z_min}:{z_max}]")
        print(f"  裁剪后尺寸: {crop_size}")
        
        # 定义输出路径，保持原文件结构
        ct_filename = os.path.basename(ct_path)
        mri_filename = os.path.basename(mri_path)
        label_filename = os.path.basename(label_path)
        
        ct_output = os.path.join(output_base, "volumes_center", ct_filename)
        mri_output = os.path.join(output_base, "volumes_center", mri_filename)
        label_output = os.path.join(output_base, "seg_center", label_filename)
        
        # 裁剪CT
        print("  裁剪CT...")
        if not crop_nifti_by_bbox(ct_path, bbox, ct_output, is_label=False):
            return False
        
        # 裁剪MRI
        print("  裁剪MRI...")
        if not crop_nifti_by_bbox(mri_path, bbox, mri_output, is_label=False):
            return False
        
        # 裁剪Label
        print("  裁剪Label...")
        if not crop_nifti_by_bbox(label_path, bbox, label_output, is_label=True):
            return False
        
        # 简单验证label文件是否成功保存
        print("  ✓ Label裁剪完成")
        
        print(f"  ✓ 患者 {patient_id} 处理完成")
        print(f"    CT: {ct_filename}")
        print(f"    MRI: {mri_filename}")
        print(f"    Label: {label_filename}")
        print()
        
        return True
        
    except Exception as e:
        print(f"  ✗ 患者 {patient_id} 处理失败: {e}")
        print()
        return False

def find_matching_files(volumes_folder, label_folder):
    """
    查找匹配的CT、MRI、Label文件
    
    Args:
        volumes_folder: CT和MRI文件夹路径（volumes_center）
        label_folder: Label文件夹路径（seg_center）
    
    Returns:
        list: 匹配的文件三元组列表 (ct_path, mri_path, label_path, patient_id)
    """
    # 获取所有文件
    volume_files = {f.stem.replace('.nii', ''): f for f in Path(volumes_folder).glob("*.nii.gz")}
    label_files = {f.stem.replace('.nii', ''): f for f in Path(label_folder).glob("*.nii.gz")}
    
    matches = []
    
    # 基于label文件查找匹配
    for label_name, label_path in label_files.items():
        # 患者ID就是label文件名（如 1PA001）
        patient_id = label_name
        
        # 在volumes_folder中查找对应的CT和MRI文件
        # 命名规则: {patient_id}_ct.nii.gz 和 {patient_id}_mr.nii.gz
        ct_name = f"{patient_id}_ct"
        mri_name = f"{patient_id}_mr"
        
        ct_match = volume_files.get(ct_name)
        mri_match = volume_files.get(mri_name)
        
        if ct_match and mri_match:
            matches.append((str(ct_match), str(mri_match), str(label_path), patient_id))
        else:
            print(f"警告: 患者 {patient_id} 缺少匹配文件")
            if not ct_match:
                print(f"  缺少CT文件: {ct_name}.nii.gz")
            if not mri_match:
                print(f"  缺少MRI文件: {mri_name}.nii.gz")
    
    return matches

def main():
    """主函数"""
    # 输入文件夹路径 - 适配pelvis数据结构
    # volumes_center文件夹包含CT和MRI文件（命名为patient_id_ct.nii.gz和patient_id_mr.nii.gz）
    # seg_center文件夹包含label文件（命名为patient_id.nii.gz）
    volumes_folder = r".\data\Public\volumes_73"
    label_folder = r".\data\Public\HN_seg_73"
    
    # 输出文件夹路径
    output_base = r".\data\Public\cropped_by_label"
    
    # 边界扩展像素数
    margin = 10  # 可以调整这个值
    
    print("基于Label边界的自适应裁剪工具 - Pelvis数据集")
    print("=" * 60)
    print(f"Volumes文件夹(CT+MRI): {volumes_folder}")
    print(f"Label文件夹: {label_folder}")
    print(f"输出文件夹: {output_base}")
    print(f"边界扩展: {margin} 像素")
    print("=" * 60)
    print()
    
    # 查找匹配的文件
    print("查找匹配的文件...")
    matches = find_matching_files(volumes_folder, label_folder)
    
    if not matches:
        print("没有找到匹配的文件!")
        return
    
    print(f"找到 {len(matches)} 套匹配的数据")
    print()
    
    # 处理每套数据
    success_count = 0
    
    for ct_path, mri_path, label_path, patient_id in matches:
        if process_patient_set(ct_path, mri_path, label_path, output_base, patient_id, margin):
            success_count += 1
    
    print("=" * 60)
    print(f"处理完成: {success_count}/{len(matches)} 套数据成功")
    print(f"输出目录结构:")
    print(f"  {output_base}/")
    print(f"  ├── volumes_center/ (CT和MRI文件)")
    print(f"  └── seg_center/ (Label文件)")
    print()
    
    if success_count < len(matches):
        print("部分数据处理失败，请检查:")
        print("1. 文件路径是否正确")
        print("2. Label文件是否包含有效标签")
        print("3. 磁盘空间是否充足")

if __name__ == "__main__":
    main()