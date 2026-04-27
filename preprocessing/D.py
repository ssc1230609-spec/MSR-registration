"""
D：Manual Inspection
医学图像伪影检测工具
用于检测CT和MRI图像中的各种伪影
常见伪影类型：
CT: 金属伪影、运动伪影、环形伪影、截断伪影
MRI: 运动伪影、化学位移伪影、混叠伪影、偏移场伪影、Gibbs振铃
********:伪影只是初步筛选，后续还需要人工逐一筛查*******
"""

import os
import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt
from scipy import ndimage
from scipy.fft import fftn, fftshift
from pathlib import Path
import json
from datetime import datetime


class NumpyEncoder(json.JSONEncoder):
    """自定义JSON编码器，处理numpy数据类型"""
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(NumpyEncoder, self).default(obj)


class ArtifactDetector:
    """医学图像伪影检测器"""
    
    def __init__(self, modality='CT'):
        """
        初始化检测器
        Args:
            modality: 'CT' or 'MRI'
        """
        self.modality = modality
        self.results = {}
        
    def load_image(self, image_path):
        """加载NIfTI图像"""
        nii = nib.load(image_path)
        data = nii.get_fdata()
        return data, nii
    
    def check_intensity_range(self, data):
        """检查强度范围异常"""
        min_val = np.min(data)
        max_val = np.max(data)
        mean_val = np.mean(data)
        std_val = np.std(data)
        
        # 检查异常值
        q1, q99 = np.percentile(data, [1, 99])
        outlier_low = min_val < (q1 - 3 * std_val)
        outlier_high = max_val > (q99 + 3 * std_val)
        
        return {
            'min': float(min_val),
            'max': float(max_val),
            'mean': float(mean_val),
            'std': float(std_val),
            'q1': float(q1),
            'q99': float(q99),
            'has_outliers': bool(outlier_low or outlier_high),
            'outlier_details': {
                'low_outliers': bool(outlier_low),
                'high_outliers': bool(outlier_high)
            }
        }
    
    def detect_metal_artifacts(self, data):
        """
        检测金属伪影（主要用于CT）
        金属会产生极端的HU值和星状伪影
        """
        if self.modality != 'CT':
            return {'applicable': False}
        
        # CT中金属通常表现为极高的HU值（>3000）
        metal_threshold_high = 3000
        metal_threshold_low = -1000  # 金属周围的暗条纹
        
        high_intensity_voxels = np.sum(data > metal_threshold_high)
        low_intensity_voxels = np.sum(data < metal_threshold_low)
        total_voxels = data.size
        
        high_ratio = high_intensity_voxels / total_voxels
        low_ratio = low_intensity_voxels / total_voxels
        
        # 检测星状伪影模式
        has_metal = high_ratio > 0.0001  # 0.01%的体素超过阈值
        
        return {
            'applicable': True,
            'suspected_metal': bool(has_metal),
            'high_intensity_ratio': float(high_ratio),
            'low_intensity_ratio': float(low_ratio),
            'high_intensity_voxels': int(high_intensity_voxels),
            'severity': 'high' if high_ratio > 0.001 else ('medium' if high_ratio > 0.0001 else 'low')
        }
    
    def detect_motion_artifacts(self, data):
        """
        检测运动伪影
        通过频域分析检测周期性模式
        """
        # 取中间切片进行分析
        mid_slice = data.shape[2] // 2
        slice_data = data[:, :, mid_slice]
        
        # FFT分析
        fft_data = np.abs(fftshift(fftn(slice_data)))
        
        # 检查高频能量
        center = np.array(fft_data.shape) // 2
        y, x = np.ogrid[:fft_data.shape[0], :fft_data.shape[1]]
        distance_from_center = np.sqrt((x - center[1])**2 + (y - center[0])**2)
        
        # 计算不同频率区域的能量
        total_energy = np.sum(fft_data**2)
        high_freq_mask = distance_from_center > (min(fft_data.shape) * 0.3)
        high_freq_energy = np.sum(fft_data[high_freq_mask]**2)
        
        high_freq_ratio = high_freq_energy / (total_energy + 1e-10)
        
        # 运动伪影通常表现为高频区域有异常的条纹
        motion_suspected = high_freq_ratio > 0.15
        
        return {
            'suspected_motion': bool(motion_suspected),
            'high_freq_energy_ratio': float(high_freq_ratio),
            'severity': 'high' if high_freq_ratio > 0.25 else ('medium' if high_freq_ratio > 0.15 else 'low')
        }
    
    def detect_truncation_artifacts(self, data):
        """
        检测截断伪影
        检查图像边缘是否有非零值（可能被截断）
        """
        # 检查所有六个面
        edges = {
            'x_min': data[0, :, :],
            'x_max': data[-1, :, :],
            'y_min': data[:, 0, :],
            'y_max': data[:, -1, :],
            'z_min': data[:, :, 0],
            'z_max': data[:, :, -1]
        }
        
        truncation_detected = {}
        for edge_name, edge_data in edges.items():
            # 如果边缘有显著的非零值，可能存在截断
            edge_mean = np.mean(np.abs(edge_data))
            global_mean = np.mean(np.abs(data))
            
            # 边缘强度超过全局平均的10%
            truncation_detected[edge_name] = bool(edge_mean > global_mean * 0.1)
        
        any_truncation = any(truncation_detected.values())
        
        return {
            'suspected_truncation': bool(any_truncation),
            'edges': truncation_detected,
            'severity': 'high' if sum(truncation_detected.values()) > 2 else 'medium'
        }
    
    def detect_bias_field(self, data):
        """
        检测偏移场（主要用于MRI）
        通过计算图像的低频变化
        """
        if self.modality != 'MRI':
            return {'applicable': False}
        
        # 取中间切片
        mid_slice = data.shape[2] // 2
        slice_data = data[:, :, mid_slice]
        
        # 高斯平滑提取低频成分
        smoothed = ndimage.gaussian_filter(slice_data, sigma=20)
        
        # 计算偏移场的变化程度
        if np.max(smoothed) > 0:
            bias_variation = np.std(smoothed) / (np.mean(smoothed) + 1e-10)
        else:
            bias_variation = 0
        
        bias_suspected = bias_variation > 0.3
        
        return {
            'applicable': True,
            'suspected_bias_field': bool(bias_suspected),
            'variation_coefficient': float(bias_variation),
            'severity': 'high' if bias_variation > 0.5 else ('medium' if bias_variation > 0.3 else 'low')
        }
    
    def detect_noise_level(self, data):
        """
        检测噪声水平
        使用背景区域估计噪声
        """
        # 假设四个角落区域是背景
        corner_size = min(data.shape) // 10
        corners = [
            data[:corner_size, :corner_size, :corner_size],
            data[-corner_size:, :corner_size, :corner_size],
            data[:corner_size, -corner_size:, :corner_size],
            data[:corner_size, :corner_size, -corner_size:]
        ]
        
        noise_estimates = []
        for corner in corners:
            noise_estimates.append(np.std(corner))
        
        avg_noise = np.mean(noise_estimates)
        signal = np.mean(np.abs(data))
        
        if signal > 0:
            snr = signal / (avg_noise + 1e-10)
        else:
            snr = 0
        
        noisy = snr < 10  # 信噪比低于10认为噪声较大
        
        return {
            'estimated_noise': float(avg_noise),
            'signal_level': float(signal),
            'snr': float(snr),
            'high_noise': bool(noisy),
            'quality': 'poor' if snr < 10 else ('fair' if snr < 20 else 'good')
        }
    
    def detect_ring_artifacts(self, data):
        """
        检测环形伪影（主要用于CT）
        """
        if self.modality != 'CT':
            return {'applicable': False}
        
        # 取中间切片
        mid_slice = data.shape[2] // 2
        slice_data = data[:, :, mid_slice]
        
        # 转换到极坐标分析同心圆模式
        center = np.array(slice_data.shape) // 2
        y, x = np.ogrid[:slice_data.shape[0], :slice_data.shape[1]]
        r = np.sqrt((x - center[1])**2 + (y - center[0])**2).astype(int)
        
        # 计算径向方差
        max_r = min(center)
        radial_profile = []
        for radius in range(0, max_r, 5):
            mask = (r >= radius) & (r < radius + 5)
            if np.any(mask):
                radial_profile.append(np.std(slice_data[mask]))
        
        if len(radial_profile) > 0:
            radial_variation = np.std(radial_profile) / (np.mean(radial_profile) + 1e-10)
        else:
            radial_variation = 0
        
        ring_suspected = radial_variation > 0.5
        
        return {
            'applicable': True,
            'suspected_rings': bool(ring_suspected),
            'radial_variation': float(radial_variation),
            'severity': 'high' if radial_variation > 1.0 else ('medium' if radial_variation > 0.5 else 'low')
        }
    
    def check_image_completeness(self, data):
        """检查图像完整性"""
        # 检查是否有大量的零值或常数值
        zero_ratio = np.sum(data == 0) / data.size
        
        # 检查是否有NaN或Inf
        has_nan = np.any(np.isnan(data))
        has_inf = np.any(np.isinf(data))
        
        # 检查数据的有效性
        unique_values = len(np.unique(data))
        
        return {
            'zero_ratio': float(zero_ratio),
            'has_nan': bool(has_nan),
            'has_inf': bool(has_inf),
            'unique_values': int(unique_values),
            'is_constant': bool(unique_values < 10),
            'excessive_zeros': bool(zero_ratio > 0.5)
        }
    
    def generate_visualization(self, data, output_path, findings):
        """生成可视化报告"""
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        fig.suptitle(f'{self.modality} Image Quality Check - {Path(output_path).stem}', fontsize=16)
        
        # 三个正交切片
        mid_x = data.shape[0] // 2
        mid_y = data.shape[1] // 2
        mid_z = data.shape[2] // 2
        
        # Axial
        axes[0, 0].imshow(data[:, :, mid_z].T, cmap='gray', origin='lower')
        axes[0, 0].set_title('Axial View')
        axes[0, 0].axis('off')
        
        # Sagittal
        axes[0, 1].imshow(data[mid_x, :, :].T, cmap='gray', origin='lower')
        axes[0, 1].set_title('Sagittal View')
        axes[0, 1].axis('off')
        
        # Coronal
        axes[0, 2].imshow(data[:, mid_y, :].T, cmap='gray', origin='lower')
        axes[0, 2].set_title('Coronal View')
        axes[0, 2].axis('off')
        
        # 强度直方图
        axes[1, 0].hist(data.flatten(), bins=100, edgecolor='black', alpha=0.7)
        axes[1, 0].set_title('Intensity Histogram')
        axes[1, 0].set_xlabel('Intensity')
        axes[1, 0].set_ylabel('Frequency')
        axes[1, 0].grid(True, alpha=0.3)
        
        # FFT频谱（用于检测周期性伪影）
        slice_data = data[:, :, mid_z]
        fft_data = np.abs(fftshift(fftn(slice_data)))
        axes[1, 1].imshow(np.log(fft_data + 1), cmap='hot')
        axes[1, 1].set_title('FFT Spectrum (Log Scale)')
        axes[1, 1].axis('off')
        
        # 检测结果摘要
        axes[1, 2].axis('off')
        summary_text = "Artifact Detection Summary:\n\n"
        
        # 添加关键发现
        if findings.get('metal_artifacts', {}).get('suspected_metal'):
            summary_text += "⚠ Metal artifacts detected\n"
        if findings.get('motion_artifacts', {}).get('suspected_motion'):
            summary_text += "⚠ Motion artifacts suspected\n"
        if findings.get('truncation', {}).get('suspected_truncation'):
            summary_text += "⚠ Truncation artifacts detected\n"
        if findings.get('bias_field', {}).get('suspected_bias_field'):
            summary_text += "⚠ Bias field detected (MRI)\n"
        if findings.get('ring_artifacts', {}).get('suspected_rings'):
            summary_text += "⚠ Ring artifacts detected\n"
        if findings.get('noise', {}).get('high_noise'):
            summary_text += "⚠ High noise level\n"
        if findings.get('completeness', {}).get('excessive_zeros'):
            summary_text += "⚠ Excessive zero values\n"
        
        if "⚠" not in summary_text:
            summary_text += "✓ No major artifacts detected"
        
        # 添加SNR信息
        snr = findings.get('noise', {}).get('snr', 0)
        summary_text += f"\n\nSNR: {snr:.2f}\n"
        summary_text += f"Quality: {findings.get('noise', {}).get('quality', 'unknown')}"
        
        axes[1, 2].text(0.1, 0.5, summary_text, fontsize=10, verticalalignment='center',
                       family='monospace', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        plt.tight_layout()
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()
    
    def analyze(self, image_path, output_dir=None):
        """
        完整分析单个图像
        Args:
            image_path: 图像文件路径
            output_dir: 输出目录
        """
        print(f"Analyzing: {image_path}")
        
        # 加载图像
        data, nii = self.load_image(image_path)
        
        # 执行所有检测
        findings = {
            'file': str(image_path),
            'modality': self.modality,
            'shape': data.shape,
            'spacing': nii.header.get_zooms()[:3],
            'intensity_range': self.check_intensity_range(data),
            'metal_artifacts': self.detect_metal_artifacts(data),
            'motion_artifacts': self.detect_motion_artifacts(data),
            'truncation': self.detect_truncation_artifacts(data),
            'bias_field': self.detect_bias_field(data),
            'ring_artifacts': self.detect_ring_artifacts(data),
            'noise': self.detect_noise_level(data),
            'completeness': self.check_image_completeness(data)
        }
        
        # 生成可视化
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
            vis_path = os.path.join(output_dir, f"{Path(image_path).stem}_artifact_check.png")
            self.generate_visualization(data, vis_path, findings)
            findings['visualization'] = vis_path
        
        return findings


def analyze_directory(ct_dir, mri_dir, output_dir):
    """
    批量分析CT和MRI目录
    Args:
        ct_dir: CT图像目录
        mri_dir: MRI图像目录
        output_dir: 输出目录
    """
    os.makedirs(output_dir, exist_ok=True)
    
    all_results = {
        'timestamp': datetime.now().isoformat(),
        'ct_results': [],
        'mri_results': []
    }
    
    # 分析CT图像
    if os.path.exists(ct_dir):
        print("\n" + "="*60)
        print("Analyzing CT images...")
        print("="*60)
        
        ct_detector = ArtifactDetector(modality='CT')
        ct_files = sorted(Path(ct_dir).glob('*.nii.gz'))
        
        ct_output_dir = os.path.join(output_dir, 'CT_checks')
        os.makedirs(ct_output_dir, exist_ok=True)
        
        for ct_file in ct_files:
            try:
                result = ct_detector.analyze(str(ct_file), ct_output_dir)
                all_results['ct_results'].append(result)
                
                # 打印关键发现
                print(f"\n{ct_file.name}:")
                if result['metal_artifacts'].get('suspected_metal'):
                    print(f"  ⚠ Metal artifacts: {result['metal_artifacts']['severity']}")
                if result['motion_artifacts'].get('suspected_motion'):
                    print(f"  ⚠ Motion artifacts: {result['motion_artifacts']['severity']}")
                if result['ring_artifacts'].get('suspected_rings'):
                    print(f"  ⚠ Ring artifacts: {result['ring_artifacts']['severity']}")
                print(f"  SNR: {result['noise']['snr']:.2f} ({result['noise']['quality']})")
                
            except Exception as e:
                print(f"Error processing {ct_file}: {e}")
    
    # 分析MRI图像
    if os.path.exists(mri_dir):
        print("\n" + "="*60)
        print("Analyzing MRI images...")
        print("="*60)
        
        mri_detector = ArtifactDetector(modality='MRI')
        mri_files = sorted(Path(mri_dir).glob('*.nii.gz'))
        
        mri_output_dir = os.path.join(output_dir, 'MRI_checks')
        os.makedirs(mri_output_dir, exist_ok=True)
        
        for mri_file in mri_files:
            try:
                result = mri_detector.analyze(str(mri_file), mri_output_dir)
                all_results['mri_results'].append(result)
                
                # 打印关键发现
                print(f"\n{mri_file.name}:")
                if result['motion_artifacts'].get('suspected_motion'):
                    print(f"  ⚠ Motion artifacts: {result['motion_artifacts']['severity']}")
                if result['bias_field'].get('suspected_bias_field'):
                    print(f"  ⚠ Bias field: {result['bias_field']['severity']}")
                print(f"  SNR: {result['noise']['snr']:.2f} ({result['noise']['quality']})")
                
            except Exception as e:
                print(f"Error processing {mri_file}: {e}")
    
    # 保存完整报告
    report_path = os.path.join(output_dir, 'artifact_detection_report.json')
    with open(report_path, 'w', encoding='utf-8') as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False, cls=NumpyEncoder)
    
    print("\n" + "="*60)
    print(f"Analysis complete! Report saved to: {report_path}")
    print("="*60)
    
    # 生成统计摘要
    generate_summary(all_results, output_dir)
    
    return all_results


def generate_summary(results, output_dir):
    """生成统计摘要"""
    summary_path = os.path.join(output_dir, 'summary.txt')
    
    with open(summary_path, 'w', encoding='utf-8') as f:
        f.write("="*80 + "\n")
        f.write("医学图像伪影检测摘要报告\n")
        f.write("="*80 + "\n\n")
        f.write(f"生成时间: {results['timestamp']}\n\n")
        
        # CT摘要
        if results['ct_results']:
            f.write("-"*80 + "\n")
            f.write(f"CT图像分析 (总计: {len(results['ct_results'])} 个文件)\n")
            f.write("-"*80 + "\n")
            
            metal_count = sum(1 for r in results['ct_results'] 
                            if r['metal_artifacts'].get('suspected_metal'))
            motion_count = sum(1 for r in results['ct_results'] 
                             if r['motion_artifacts'].get('suspected_motion'))
            ring_count = sum(1 for r in results['ct_results'] 
                           if r['ring_artifacts'].get('suspected_rings'))
            high_noise_count = sum(1 for r in results['ct_results'] 
                                 if r['noise'].get('high_noise'))
            
            f.write(f"检测到金属伪影: {metal_count} 个文件\n")
            f.write(f"检测到运动伪影: {motion_count} 个文件\n")
            f.write(f"检测到环形伪影: {ring_count} 个文件\n")
            f.write(f"高噪声图像: {high_noise_count} 个文件\n\n")
            
            # SNR统计
            snr_values = [r['noise']['snr'] for r in results['ct_results']]
            f.write(f"信噪比 (SNR) 统计:\n")
            f.write(f"  平均值: {np.mean(snr_values):.2f}\n")
            f.write(f"  中位数: {np.median(snr_values):.2f}\n")
            f.write(f"  最小值: {np.min(snr_values):.2f}\n")
            f.write(f"  最大值: {np.max(snr_values):.2f}\n\n")
        
        # MRI摘要
        if results['mri_results']:
            f.write("-"*80 + "\n")
            f.write(f"MRI图像分析 (总计: {len(results['mri_results'])} 个文件)\n")
            f.write("-"*80 + "\n")
            
            motion_count = sum(1 for r in results['mri_results'] 
                             if r['motion_artifacts'].get('suspected_motion'))
            bias_count = sum(1 for r in results['mri_results'] 
                           if r['bias_field'].get('suspected_bias_field'))
            high_noise_count = sum(1 for r in results['mri_results'] 
                                 if r['noise'].get('high_noise'))
            
            f.write(f"检测到运动伪影: {motion_count} 个文件\n")
            f.write(f"检测到偏移场: {bias_count} 个文件\n")
            f.write(f"高噪声图像: {high_noise_count} 个文件\n\n")
            
            # SNR统计
            snr_values = [r['noise']['snr'] for r in results['mri_results']]
            f.write(f"信噪比 (SNR) 统计:\n")
            f.write(f"  平均值: {np.mean(snr_values):.2f}\n")
            f.write(f"  中位数: {np.median(snr_values):.2f}\n")
            f.write(f"  最小值: {np.min(snr_values):.2f}\n")
            f.write(f"  最大值: {np.max(snr_values):.2f}\n\n")
        
        f.write("="*80 + "\n")
        f.write("详细结果请查看: artifact_detection_report.json\n")
        f.write("可视化结果保存在相应的子目录中\n")
        f.write("="*80 + "\n")
    
    print(f"\nSummary saved to: {summary_path}")


def main():
    """主函数"""
    import argparse
    
    parser = argparse.ArgumentParser(description='医学图像伪影检测工具')
    parser.add_argument('--ct_dir', type=str, 
                       default=r'./data/xxxCT',
                       help='CT图像目录')
    parser.add_argument('--mri_dir', type=str, 
                       default=r'./data/xxxMRI',
                       help='MRI图像目录')
    parser.add_argument('--output_dir', type=str, 
                       default=r'./data/artifact_check_results',
                       help='输出目录')
    parser.add_argument('--single_file', type=str, 
                       help='分析单个文件 (指定模态: --modality CT/MRI)')
    parser.add_argument('--modality', type=str, choices=['CT', 'MRI'], 
                       default='CT',
                       help='单个文件的模态类型')
    
    args = parser.parse_args()
    
    if args.single_file:
        # 分析单个文件
        detector = ArtifactDetector(modality=args.modality)
        os.makedirs(args.output_dir, exist_ok=True)
        result = detector.analyze(args.single_file, args.output_dir)
        
        # 打印结果
        print("\n" + "="*60)
        print("Analysis Results:")
        print("="*60)
        print(json.dumps(result, indent=2, ensure_ascii=False, cls=NumpyEncoder))
        
    else:
        # 批量分析目录
        analyze_directory(args.ct_dir, args.mri_dir, args.output_dir)


if __name__ == '__main__':
    main()
