"""
B1: Preliminary cropping
"""

import os
import nibabel as nib
import numpy as np

def crop_nifti_x_axis(input_folder, output_folder, x_start=140, x_end=400):
    """
    裁剪文件夹中所有nii.gz文件的X轴
    
    参数:
    input_folder: 输入文件夹路径
    output_folder: 输出文件夹路径
    x_start: X轴起始位置
    x_end: X轴结束位置
    """
    # 确保输出文件夹存在
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    
    # 获取所有nii.gz文件
    nii_files = [f for f in os.listdir(input_folder) if f.endswith('.nii.gz')]
    
    for file_name in nii_files:
        try:
            # 构建完整的文件路径
            input_path = os.path.join(input_folder, file_name)
            output_path = os.path.join(output_folder, file_name)
            
            # 加载nii.gz文件
            img = nib.load(input_path)
            data = img.get_fdata()
            
            # 获取图像的原始信息
            affine = img.affine
            header = img.header
            
            # 裁剪X轴
            cropped_data = data[x_start:x_end, :, :]
            
            # 创建新的nifti图像
            new_img = nib.Nifti1Image(cropped_data, affine, header)
            
            # 保存裁剪后的图像
            nib.save(new_img, output_path)
            
            print(f"成功处理文件: {file_name}")
            
        except Exception as e:
            print(f"处理文件 {file_name} 时出错: {str(e)}")

if __name__ == "__main__":
    # 设置输入和输出文件夹路径
    input_folder = r"./data/xxxCT"
    output_folder = r"./data/crop_xxxCTnifit"
    
    # 执行裁剪操作
    crop_nifti_x_axis(input_folder, output_folder)
    
    print("处理完成！")