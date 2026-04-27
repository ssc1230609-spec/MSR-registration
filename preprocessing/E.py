"""
E：Pre-Registration
"""
import ants
import numpy as np
import os
import re
import argparse
import traceback


def extract_id(filename):
    # 提取 ID
    match = re.search(r'Spine_(\d+)_', filename)
    if match:
        return match.group(1)
    match = re.search(r'c([\d-]+)_', filename)
    if match:
        return match.group(1)  # 返回数字部分（包括可能的-符号）
    return None
def register_mri_to_ct(ct_path, mri_path, output_dir=None, output_path=None):
    ct_img = ants.image_read(ct_path)   # 固定图像
    mri_img = ants.image_read(mri_path) # 移动图像
    
    # 使用 ANTs 的原生 intensity normalization（更适合 MI）
    ct_norm = ants.iMath_normalize(ct_img)
    mri_norm = ants.iMath_normalize(mri_img)
    
    print(f"开始配准: {os.path.basename(mri_path)} 到 {os.path.basename(ct_path)}...")
    # ANTs会自动进行初始中心对齐，无需手动设置initial_transform
    registration_result = ants.registration(
        fixed=ct_norm,     # CT作为固定图像
        moving=mri_norm,   # MRI作为移动图像
        type_of_transform='Rigid',  # 刚性配准
        aff_metric='MI',   # 互信息度量
        aff_sampling=32,   # 多点MI采样，适用于医学图像
        aff_iterations=(160, 80, 40),  # 标准的刚性配准迭代次数
        aff_smoothing_sigmas=(2, 1, 0),  # 对应的平滑参数
        aff_shrink_factors=(2, 1, 1)   # 对应的重采样因子
    )
    # 计算配准后的互信息度量值
    mi_value = ants.image_mutual_information(ct_norm, ants.apply_transforms(
        fixed=ct_norm,
        moving=mri_norm,
        transformlist=registration_result['fwdtransforms']
    ))
    # 应用变换到原始图像（保持原始强度范围）
    mri_registered = ants.apply_transforms(
        fixed=ct_img,
        moving=mri_img,
        transformlist=registration_result['fwdtransforms'],
        interpolator="bSpline"
    )
    
    # 使用MRI文件名作为输出文件名
    # 生成输出路径（支持 output_path 或 output_dir）
    if output_path is None:
        if output_dir is None:
            raise ValueError("请提供 output_dir 或 output_path")
        output_name = os.path.basename(mri_path)
        output_path = os.path.join(output_dir, output_name)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    ants.image_write(mri_registered, output_path)
    print(f"配准完成，互信息度量值: {mi_value:.4f}，结果保存至: {output_path}")

if __name__ == '__main__':
    # 默认目录（可被命令行参数覆盖）
    ct_dir = './data/cropped_by_label_mri/CT'
    mri_dir = './data/cropped_by_label_mri/MRI'
    output_dir = './data/cropped_by_label_mri/rigid-mri'
    
    parser = argparse.ArgumentParser(description='MRI刚性配准到CT：支持单文件或批量')
    parser.add_argument('--ct', type=str, help='CT固定图像路径')
    parser.add_argument('--mri', type=str, help='MRI移动图像路径')
    parser.add_argument('--output', type=str, help='输出文件路径（包含文件名）')
    parser.add_argument('--output-dir', type=str, help='输出目录（单文件模式）')
    parser.add_argument('--ct-dir', type=str, default=ct_dir, help='CT目录（批量模式）')
    parser.add_argument('--mri-dir', type=str, default=mri_dir, help='MRI目录（批量模式）')
    parser.add_argument('--dataset-output-dir', type=str, default=output_dir, help='批量输出目录')
    args = parser.parse_args()

    # 单文件模式
    if args.ct and args.mri:
        try:
            register_mri_to_ct(args.ct, args.mri, output_dir=args.output_dir, output_path=args.output)
        except Exception as e:
            print(f"处理单文件时发生错误: {str(e)}")
            print(f"详细错误信息: {traceback.format_exc()}")
        exit(0)

    # 批量模式
    ct_dir = args.ct_dir
    mri_dir = args.mri_dir
    output_dir = args.dataset_output_dir
    os.makedirs(output_dir, exist_ok=True)
    
    # 获取所有CT和MRI文件
    ct_files = [f for f in os.listdir(ct_dir) if f.endswith('.nii.gz')]
    mri_files = [f for f in os.listdir(mri_dir) if f.endswith('.nii.gz')]
    
    # 根据文件名中的ID匹配CT和MRI
    for ct_file in ct_files:
        ct_id = extract_id(ct_file)
        if not ct_id:
            print(f"无法从CT文件名中提取ID: {ct_file}")
            continue
            
        # 查找对应的MRI文件
        matching_mri = [mri for mri in mri_files if extract_id(mri) == ct_id]
        
        # 如果没有找到匹配的MRI文件，打印调试信息
        if not matching_mri:
            print(f"未找到与CT文件 {ct_file} (ID: {ct_id}) 匹配的MRI文件")
        
        if matching_mri:
            for mri_file in matching_mri:
                ct_path = os.path.join(ct_dir, ct_file)
                mri_path = os.path.join(mri_dir, mri_file)
                
                try:
                    register_mri_to_ct(ct_path, mri_path, output_dir)
                except Exception as e:
                    print(f"处理 {ct_file} 和 {mri_file} 时发生错误: {str(e)}")
                    print(f"详细错误信息: {traceback.format_exc()}")
        else:
            print(f"未找到与 {ct_file} 匹配的MRI文件")
    
    print("所有配准任务完成！")