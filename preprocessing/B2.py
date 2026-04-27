"""
B2: Preliminary cropping
"""

import nibabel as nib
import numpy as np
import os
from pathlib import Path

def crop_ct_z_axis(input_path, output_path, z_start=164, z_end=378):
    """
    对CT图像在Z轴方向进行裁剪
    
    参数:
    input_path: 输入NIfTI文件路径
    output_path: 输出NIfTI文件路径
    z_start: Z轴起始位置（默认164）
    z_end: Z轴结束位置（默认378）
    """
    try:
        # 加载NIfTI文件
        img = nib.load(input_path)
        data = img.get_fdata()
        
        print(f"原始图像形状: {data.shape}")
        
        # 检查Z轴范围
        if z_end > data.shape[2]:
            print(f"警告: z_end ({z_end}) 超过图像Z轴大小 ({data.shape[2]})，将调整为 {data.shape[2]}")
            z_end = data.shape[2]
        
        if z_start < 0:
            print(f"警告: z_start ({z_start}) 小于0，将调整为0")
            z_start = 0
            
        if z_start >= z_end:
            raise ValueError(f"z_start ({z_start}) 必须小于 z_end ({z_end})")
        
        # 在Z轴方向裁剪
        cropped_data = data[:, :, z_start:z_end]
        print(f"裁剪后图像形状: {cropped_data.shape}")
        
        # 更新仿射矩阵以反映Z轴偏移
        affine = img.affine.copy()
        # 调整Z轴的平移部分
        affine[2, 3] += z_start * affine[2, 2]
        
        # 创建新的NIfTI图像
        cropped_img = nib.Nifti1Image(cropped_data, affine, img.header)
        
        # 确保输出目录存在
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        # 保存裁剪后的图像
        nib.save(cropped_img, output_path)
        print(f"成功保存裁剪后的图像到: {output_path}")
        
        return True
        
    except Exception as e:
        print(f"处理文件 {input_path} 时出错: {str(e)}")
        return False

def batch_crop_ct_files(input_dir, output_dir, z_start=164, z_end=378):
    """
    批量处理目录中的CT文件
    
    参数:
    input_dir: 输入目录路径
    output_dir: 输出目录路径
    z_start: Z轴起始位置（默认164）
    z_end: Z轴结束位置（默认378）
    """
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    
    # 创建输出目录
    output_path.mkdir(parents=True, exist_ok=True)
    
    # 查找所有NIfTI文件
    nifti_files = []
    for ext in ['*.nii', '*.nii.gz']:
        nifti_files.extend(input_path.glob(ext))
    
    if not nifti_files:
        print(f"在目录 {input_dir} 中未找到NIfTI文件")
        return
    
    print(f"找到 {len(nifti_files)} 个NIfTI文件")
    
    success_count = 0
    for file_path in nifti_files:
        print(f"\n处理文件: {file_path.name}")
        
        # 构建输出文件路径
        output_file = output_path / f"cropped_{file_path.name}"
        
        # 处理文件
        if crop_ct_z_axis(str(file_path), str(output_file), z_start, z_end):
            success_count += 1
    
    print(f"\n批量处理完成！成功处理 {success_count}/{len(nifti_files)} 个文件")

if __name__ == "__main__":
    # 直接指定路径进行处理
    
    # 请根据您的实际路径修改以下路径
    input_file = r"./data/standardized/mri_standardized.nii.gz"  # 修改为您的CT文件路径
    output_file = r"./data/crop/mri_cropped.nii.gz"   # 修改为您想要的输出路径


    

    
    print("请根据需要修改文件路径并取消相应代码的注释后运行")