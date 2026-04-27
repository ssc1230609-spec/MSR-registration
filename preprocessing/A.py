"""
A： N4 bias field correction
N4偏置场矫正 - 用于MRI数据
使用SimpleITK的N4BiasFieldCorrectionImageFilter进行偏置场矫正
"""

import os
import SimpleITK as sitk
from pathlib import Path
from tqdm import tqdm


def n4_bias_correction(input_path, output_path, mask=None, shrink_factor=4, num_iterations=[50, 50, 50, 50]):
    """
    对MRI图像进行N4偏置场矫正
    
    Parameters:
    -----------
    input_path : str
        输入MRI图像路径
    output_path : str
        输出矫正后图像路径
    mask : SimpleITK.Image, optional
        掩码图像（可选）
    shrink_factor : int
        收缩因子，用于加速计算（默认4）
    num_iterations : list
        每个分辨率级别的迭代次数（默认[50, 50, 50, 50]）
    
    Returns:
    --------
    bool : 是否成功
    """
    try:
        # 读取输入图像
        input_image = sitk.ReadImage(input_path, sitk.sitkFloat32)
        
        # 如果没有提供mask，创建一个基于Otsu阈值的mask
        if mask is None:
            # 使用Otsu阈值自动生成mask
            mask_image = sitk.OtsuThreshold(input_image, 0, 1, 200)
        else:
            mask_image = mask
        
        # 设置收缩因子以加速计算
        if shrink_factor > 1:
            input_image = sitk.Shrink(input_image, [shrink_factor] * input_image.GetDimension())
            mask_image = sitk.Shrink(mask_image, [shrink_factor] * mask_image.GetDimension())
        
        # 创建N4偏置场矫正滤波器
        corrector = sitk.N4BiasFieldCorrectionImageFilter()
        
        # 设置参数
        corrector.SetMaximumNumberOfIterations(num_iterations)
        corrector.SetConvergenceThreshold(0.001)
        
        # 执行矫正
        corrected_image = corrector.Execute(input_image, mask_image)
        
        # 如果使用了收缩，需要恢复到原始尺寸
        if shrink_factor > 1:
            # 读取原始图像用于恢复尺寸
            original_image = sitk.ReadImage(input_path, sitk.sitkFloat32)
            
            # 获取偏置场
            log_bias_field = corrector.GetLogBiasFieldAsImage(input_image)
            
            # 将偏置场恢复到原始尺寸
            log_bias_field = sitk.Resample(log_bias_field, original_image)
            
            # 应用偏置场到原始图像
            bias_field = sitk.Exp(log_bias_field)
            corrected_image = original_image / bias_field
        
        # 保存矫正后的图像
        sitk.WriteImage(corrected_image, output_path)
        
        return True
        
    except Exception as e:
        print(f"处理 {input_path} 时出错: {str(e)}")
        return False


def process_mri_directory(input_dir, output_dir, overwrite=False):
    """
    批量处理MRI目录中的所有文件
    
    Parameters:
    -----------
    input_dir : str
        输入MRI目录路径
    output_dir : str
        输出目录路径
    overwrite : bool
        是否覆盖已存在的文件
    """
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    
    # 获取所有MRI文件
    input_path = Path(input_dir)
    mri_files = sorted(list(input_path.glob("*.nii.gz")))
    
    print(f"找到 {len(mri_files)} 个MRI文件")
    print(f"输入目录: {input_dir}")
    print(f"输出目录: {output_dir}")
    print("-" * 60)
    
    success_count = 0
    fail_count = 0
    skip_count = 0
    
    # 处理每个文件
    for mri_file in tqdm(mri_files, desc="N4偏置场矫正"):
        # 构建输出路径
        output_path = os.path.join(output_dir, mri_file.name)
        
        # 检查是否已存在
        if os.path.exists(output_path) and not overwrite:
            print(f"跳过 {mri_file.name} (已存在)")
            skip_count += 1
            continue
        
        # 执行N4矫正
        if n4_bias_correction(str(mri_file), output_path):
            success_count += 1
        else:
            fail_count += 1
    
    # 打印统计信息
    print("\n" + "=" * 60)
    print(f"处理完成!")
    print(f"成功: {success_count} 个文件")
    print(f"失败: {fail_count} 个文件")
    print(f"跳过: {skip_count} 个文件")
    print("=" * 60)


if __name__ == "__main__":
    # 设置输入输出路径
    input_mri_dir = r"./data/xxxMRI"  # 替换为你的MRI数据目录路径
    output_mri_dir = r"./data/xxxMRI_N4corrected"
    
    # 执行批量处理
    process_mri_directory(
        input_dir=input_mri_dir,
        output_dir=output_mri_dir,
        overwrite=False  # 设置为True可以覆盖已存在的文件
    )
