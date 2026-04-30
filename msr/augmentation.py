"""
数据增强模块 - 用于医学图像配准任务
确保源图像和目标图像应用相同的空间变换
"""

import numpy as np
import torch
import torch.nn.functional as F
from scipy.ndimage import rotate, shift, zoom, affine_transform


class SpatialAugmentation:
    """
    空间变换增强类
    支持旋转、翻转、缩放、平移等操作
    """
    
    def __init__(
        self,
        rotation_range=15,      # 旋转角度范围 (度)
        translation_range=0.1,  # 平移范围 (相对于图像尺寸的比例)
        scale_range=(0.9, 1.1), # 缩放范围
        flip_prob=0.5,          # 翻转概率
        apply_prob=0.8,         # 应用增强的概率
        use_torch=True,         # 是否使用torch实现(更快)
    ):
        self.rotation_range = rotation_range
        self.translation_range = translation_range
        self.scale_range = scale_range
        self.flip_prob = flip_prob
        self.apply_prob = apply_prob
        self.use_torch = use_torch
    
    def __call__(self, src_vol, tgt_vol, src_seg, tgt_seg):
        """
        对源图像和目标图像应用相同的空间变换
        
        Args:
            src_vol: 源图像 [B, H, W, D, C] 或 [B, C, H, W, D]
            tgt_vol: 目标图像 [B, H, W, D, C] 或 [B, C, H, W, D]
            src_seg: 源分割标签 [B, H, W, D, C] 或 [B, C, H, W, D]
            tgt_seg: 目标分割标签 [B, H, W, D, C] 或 [B, C, H, W, D]
        
        Returns:
            增强后的 (src_vol, tgt_vol, src_seg, tgt_seg)
        """
        # 判断是否应用增强
        if np.random.rand() > self.apply_prob:
            return src_vol, tgt_vol, src_seg, tgt_seg
        
        if self.use_torch:
            return self._augment_torch(src_vol, tgt_vol, src_seg, tgt_seg)
        else:
            return self._augment_numpy(src_vol, tgt_vol, src_seg, tgt_seg)
    
    def _augment_torch(self, src_vol, tgt_vol, src_seg, tgt_seg):
        """使用PyTorch实现的增强(更快,支持GPU)"""
        # 检查输入格式并转换为torch tensor
        is_numpy = isinstance(src_vol, np.ndarray)
        if is_numpy:
            device = 'cpu'
            src_vol = torch.from_numpy(src_vol.copy()).float()
            tgt_vol = torch.from_numpy(tgt_vol.copy()).float()
            src_seg = torch.from_numpy(src_seg.copy()).float()
            tgt_seg = torch.from_numpy(tgt_seg.copy()).float()
        else:
            device = src_vol.device
        
        # 检查数据格式: [B, H, W, D, C] -> [B, C, H, W, D]
        if src_vol.shape[-1] < src_vol.shape[1]:  # 最后一维是通道
            src_vol = src_vol.permute(0, 4, 1, 2, 3).contiguous()
            tgt_vol = tgt_vol.permute(0, 4, 1, 2, 3).contiguous()
            src_seg = src_seg.permute(0, 4, 1, 2, 3).contiguous()
            tgt_seg = tgt_seg.permute(0, 4, 1, 2, 3).contiguous()
            need_permute_back = True
        else:
            need_permute_back = False
            # 确保是连续的
            src_vol = src_vol.contiguous()
            tgt_vol = tgt_vol.contiguous()
            src_seg = src_seg.contiguous()
            tgt_seg = tgt_seg.contiguous()
        
        batch_size = src_vol.shape[0]
        
        # 创建输出tensor列表
        src_vol_list = []
        tgt_vol_list = []
        src_seg_list = []
        tgt_seg_list = []
        
        # 生成随机变换参数(对batch中的每个样本)
        for b in range(batch_size):
            # 获取当前样本
            curr_src_vol = src_vol[b:b+1]
            curr_tgt_vol = tgt_vol[b:b+1]
            curr_src_seg = src_seg[b:b+1]
            curr_tgt_seg = tgt_seg[b:b+1]
            
            # 1. 随机翻转
            if np.random.rand() < self.flip_prob:
                axis = np.random.choice([2, 3, 4])  # 随机选择一个空间轴
                curr_src_vol = torch.flip(curr_src_vol, dims=[axis])
                curr_tgt_vol = torch.flip(curr_tgt_vol, dims=[axis])
                curr_src_seg = torch.flip(curr_src_seg, dims=[axis])
                curr_tgt_seg = torch.flip(curr_tgt_seg, dims=[axis])
            
            # 2. 随机旋转 (在axial平面,即H-W平面)
            if self.rotation_range > 0:
                angle = np.random.uniform(-self.rotation_range, self.rotation_range)
                if abs(angle) > 1:  # 只有角度大于1度才旋转
                    curr_src_vol = self._rotate_3d_torch(curr_src_vol.squeeze(0), angle, mode='bilinear').unsqueeze(0)
                    curr_tgt_vol = self._rotate_3d_torch(curr_tgt_vol.squeeze(0), angle, mode='bilinear').unsqueeze(0)
                    curr_src_seg = self._rotate_3d_torch(curr_src_seg.squeeze(0), angle, mode='nearest').unsqueeze(0)
                    curr_tgt_seg = self._rotate_3d_torch(curr_tgt_seg.squeeze(0), angle, mode='nearest').unsqueeze(0)
            
            # 3. 随机缩放
            if self.scale_range != (1.0, 1.0):
                scale = np.random.uniform(self.scale_range[0], self.scale_range[1])
                if abs(scale - 1.0) > 0.01:  # 只有缩放比例变化大于1%才缩放
                    curr_src_vol = self._scale_3d_torch(curr_src_vol.squeeze(0), scale, mode='trilinear').unsqueeze(0)
                    curr_tgt_vol = self._scale_3d_torch(curr_tgt_vol.squeeze(0), scale, mode='trilinear').unsqueeze(0)
                    curr_src_seg = self._scale_3d_torch(curr_src_seg.squeeze(0), scale, mode='nearest').unsqueeze(0)
                    curr_tgt_seg = self._scale_3d_torch(curr_tgt_seg.squeeze(0), scale, mode='nearest').unsqueeze(0)
            
            # 添加到列表
            src_vol_list.append(curr_src_vol)
            tgt_vol_list.append(curr_tgt_vol)
            src_seg_list.append(curr_src_seg)
            tgt_seg_list.append(curr_tgt_seg)
        
        # 合并batch
        src_vol = torch.cat(src_vol_list, dim=0)
        tgt_vol = torch.cat(tgt_vol_list, dim=0)
        src_seg = torch.cat(src_seg_list, dim=0)
        tgt_seg = torch.cat(tgt_seg_list, dim=0)
        
        # 转换回原始格式
        if need_permute_back:
            src_vol = src_vol.permute(0, 2, 3, 4, 1).contiguous()
            tgt_vol = tgt_vol.permute(0, 2, 3, 4, 1).contiguous()
            src_seg = src_seg.permute(0, 2, 3, 4, 1).contiguous()
            tgt_seg = tgt_seg.permute(0, 2, 3, 4, 1).contiguous()
        
        if is_numpy:
            src_vol = src_vol.cpu().numpy()
            tgt_vol = tgt_vol.cpu().numpy()
            src_seg = src_seg.cpu().numpy()
            tgt_seg = tgt_seg.cpu().numpy()
        
        return src_vol, tgt_vol, src_seg, tgt_seg
    
    def _rotate_3d_torch(self, volume, angle, mode='bilinear'):
        """
        在axial平面(H-W)旋转3D体积
        volume: [C, H, W, D]
        """
        C, H, W, D = volume.shape
        device = volume.device
        
        # 创建旋转矩阵
        angle_rad = angle * np.pi / 180
        cos_a = np.cos(angle_rad)
        sin_a = np.sin(angle_rad)
        
        # 2D旋转矩阵 (在H-W平面)
        theta = torch.tensor([
            [cos_a, -sin_a, 0],
            [sin_a, cos_a, 0]
        ], dtype=torch.float32, device=device).unsqueeze(0)
        
        # 对每个深度切片应用旋转
        rotated_slices = []
        for d in range(D):
            slice_2d = volume[:, :, :, d].unsqueeze(0)  # [1, C, H, W]
            grid = F.affine_grid(theta, slice_2d.size(), align_corners=False)
            rotated_slice = F.grid_sample(slice_2d, grid, mode=mode, 
                                         padding_mode='border', align_corners=False)
            rotated_slices.append(rotated_slice.squeeze(0))
        
        return torch.stack(rotated_slices, dim=-1)  # [C, H, W, D]
    
    def _scale_3d_torch(self, volume, scale, mode='trilinear'):
        """
        缩放3D体积
        volume: [C, H, W, D]
        """
        C, H, W, D = volume.shape
        
        # 使用interpolate进行缩放
        volume_5d = volume.unsqueeze(0)  # [1, C, H, W, D]
        
        # 计算新的尺寸
        new_size = [int(H * scale), int(W * scale), int(D * scale)]
        
        # 缩放
        if mode == 'nearest':
            scaled = F.interpolate(volume_5d, size=new_size, mode='nearest')
        else:
            scaled = F.interpolate(volume_5d, size=new_size, mode=mode, align_corners=False)
        
        # 裁剪或填充回原始尺寸
        scaled = scaled.squeeze(0)  # [C, H', W', D']
        result = torch.zeros_like(volume)
        
        # 计算裁剪/填充的起始位置
        h_start = max(0, (scaled.shape[1] - H) // 2)
        w_start = max(0, (scaled.shape[2] - W) // 2)
        d_start = max(0, (scaled.shape[3] - D) // 2)
        
        h_start_orig = max(0, (H - scaled.shape[1]) // 2)
        w_start_orig = max(0, (W - scaled.shape[2]) // 2)
        d_start_orig = max(0, (D - scaled.shape[3]) // 2)
        
        h_end = min(scaled.shape[1], h_start + H)
        w_end = min(scaled.shape[2], w_start + W)
        d_end = min(scaled.shape[3], d_start + D)
        
        h_size = h_end - h_start
        w_size = w_end - w_start
        d_size = d_end - d_start
        
        result[:, h_start_orig:h_start_orig+h_size, 
               w_start_orig:w_start_orig+w_size,
               d_start_orig:d_start_orig+d_size] = scaled[:, h_start:h_end, 
                                                           w_start:w_end, 
                                                           d_start:d_end]
        
        return result
    
    def _augment_numpy(self, src_vol, tgt_vol, src_seg, tgt_seg):
        """使用NumPy/SciPy实现的增强(CPU only)"""
        # 检查数据格式
        if src_vol.shape[-1] < src_vol.shape[1]:  # [B, H, W, D, C]
            need_permute = False
        else:  # [B, C, H, W, D]
            src_vol = np.transpose(src_vol, (0, 2, 3, 4, 1))
            tgt_vol = np.transpose(tgt_vol, (0, 2, 3, 4, 1))
            src_seg = np.transpose(src_seg, (0, 2, 3, 4, 1))
            tgt_seg = np.transpose(tgt_seg, (0, 2, 3, 4, 1))
            need_permute = True
        
        batch_size = src_vol.shape[0]
        
        for b in range(batch_size):
            # 1. 随机翻转
            if np.random.rand() < self.flip_prob:
                axis = np.random.choice([0, 1, 2])  # H, W, D
                src_vol[b] = np.flip(src_vol[b], axis=axis).copy()
                tgt_vol[b] = np.flip(tgt_vol[b], axis=axis).copy()
                src_seg[b] = np.flip(src_seg[b], axis=axis).copy()
                tgt_seg[b] = np.flip(tgt_seg[b], axis=axis).copy()
            
            # 2. 随机旋转
            if self.rotation_range > 0:
                angle = np.random.uniform(-self.rotation_range, self.rotation_range)
                if abs(angle) > 1:
                    axes = (0, 1)  # 在H-W平面旋转
                    src_vol[b] = self._rotate_volume_numpy(src_vol[b], angle, axes, order=1)
                    tgt_vol[b] = self._rotate_volume_numpy(tgt_vol[b], angle, axes, order=1)
                    src_seg[b] = self._rotate_volume_numpy(src_seg[b], angle, axes, order=0)
                    tgt_seg[b] = self._rotate_volume_numpy(tgt_seg[b], angle, axes, order=0)
            
            # 3. 随机缩放
            if self.scale_range != (1.0, 1.0):
                scale = np.random.uniform(self.scale_range[0], self.scale_range[1])
                if abs(scale - 1.0) > 0.01:
                    src_vol[b] = self._scale_volume_numpy(src_vol[b], scale, order=1)
                    tgt_vol[b] = self._scale_volume_numpy(tgt_vol[b], scale, order=1)
                    src_seg[b] = self._scale_volume_numpy(src_seg[b], scale, order=0)
                    tgt_seg[b] = self._scale_volume_numpy(tgt_seg[b], scale, order=0)
        
        # 转换回原始格式
        if need_permute:
            src_vol = np.transpose(src_vol, (0, 4, 1, 2, 3))
            tgt_vol = np.transpose(tgt_vol, (0, 4, 1, 2, 3))
            src_seg = np.transpose(src_seg, (0, 4, 1, 2, 3))
            tgt_seg = np.transpose(tgt_seg, (0, 4, 1, 2, 3))
        
        return src_vol, tgt_vol, src_seg, tgt_seg
    
    def _rotate_volume_numpy(self, volume, angle, axes, order=1):
        """旋转体积 (numpy实现)"""
        # volume: [H, W, D, C]
        C = volume.shape[-1]
        rotated_channels = []
        for c in range(C):
            rotated = rotate(volume[..., c], angle, axes=axes, 
                           reshape=False, order=order, mode='nearest')
            rotated_channels.append(rotated)
        return np.stack(rotated_channels, axis=-1)
    
    def _scale_volume_numpy(self, volume, scale, order=1):
        """缩放体积 (numpy实现)"""
        # volume: [H, W, D, C]
        H, W, D, C = volume.shape
        zoom_factors = [scale, scale, scale, 1]  # 不缩放通道维度
        
        scaled = zoom(volume, zoom_factors, order=order, mode='nearest')
        
        # 裁剪或填充回原始尺寸
        result = np.zeros_like(volume)
        
        h_start = max(0, (scaled.shape[0] - H) // 2)
        w_start = max(0, (scaled.shape[1] - W) // 2)
        d_start = max(0, (scaled.shape[2] - D) // 2)
        
        h_start_orig = max(0, (H - scaled.shape[0]) // 2)
        w_start_orig = max(0, (W - scaled.shape[1]) // 2)
        d_start_orig = max(0, (D - scaled.shape[2]) // 2)
        
        h_end = min(scaled.shape[0], h_start + H)
        w_end = min(scaled.shape[1], w_start + W)
        d_end = min(scaled.shape[2], d_start + D)
        
        h_size = h_end - h_start
        w_size = w_end - w_start
        d_size = d_end - d_start
        
        result[h_start_orig:h_start_orig+h_size,
               w_start_orig:w_start_orig+w_size,
               d_start_orig:d_start_orig+d_size, :] = scaled[h_start:h_end,
                                                              w_start:w_end,
                                                              d_start:d_end, :]
        
        return result


class IntensityAugmentation:
    """
    强度增强类 (仅用于图像,不应用于分割标签)
    """
    
    def __init__(
        self,
        brightness_range=0.2,   # 亮度调整范围
        contrast_range=0.2,     # 对比度调整范围
        gamma_range=(0.8, 1.2), # 伽马校正范围
        noise_std=0.01,         # 高斯噪声标准差
        apply_prob=0.5,         # 应用增强的概率
    ):
        self.brightness_range = brightness_range
        self.contrast_range = contrast_range
        self.gamma_range = gamma_range
        self.noise_std = noise_std
        self.apply_prob = apply_prob
    
    def __call__(self, src_vol, tgt_vol):
        """
        对源图像和目标图像应用强度增强
        注意: 不对分割标签应用强度增强
        """
        if np.random.rand() > self.apply_prob:
            return src_vol, tgt_vol
        
        is_numpy = isinstance(src_vol, np.ndarray)
        
        # 亮度调整
        if self.brightness_range > 0:
            brightness = np.random.uniform(-self.brightness_range, self.brightness_range)
            src_vol = np.clip(src_vol + brightness, 0, 1) if is_numpy else torch.clamp(src_vol + brightness, 0, 1)
            tgt_vol = np.clip(tgt_vol + brightness, 0, 1) if is_numpy else torch.clamp(tgt_vol + brightness, 0, 1)
        
        # 对比度调整
        if self.contrast_range > 0:
            contrast = np.random.uniform(1 - self.contrast_range, 1 + self.contrast_range)
            if is_numpy:
                src_vol = np.clip((src_vol - 0.5) * contrast + 0.5, 0, 1)
                tgt_vol = np.clip((tgt_vol - 0.5) * contrast + 0.5, 0, 1)
            else:
                src_vol = torch.clamp((src_vol - 0.5) * contrast + 0.5, 0, 1)
                tgt_vol = torch.clamp((tgt_vol - 0.5) * contrast + 0.5, 0, 1)
        
        # 伽马校正
        if self.gamma_range != (1.0, 1.0):
            gamma = np.random.uniform(self.gamma_range[0], self.gamma_range[1])
            if is_numpy:
                src_vol = np.power(src_vol, gamma)
                tgt_vol = np.power(tgt_vol, gamma)
            else:
                src_vol = torch.pow(src_vol, gamma)
                tgt_vol = torch.pow(tgt_vol, gamma)
        
        # 添加高斯噪声
        if self.noise_std > 0:
            if is_numpy:
                noise_src = np.random.normal(0, self.noise_std, src_vol.shape)
                noise_tgt = np.random.normal(0, self.noise_std, tgt_vol.shape)
                src_vol = np.clip(src_vol + noise_src, 0, 1)
                tgt_vol = np.clip(tgt_vol + noise_tgt, 0, 1)
            else:
                noise_src = torch.randn_like(src_vol) * self.noise_std
                noise_tgt = torch.randn_like(tgt_vol) * self.noise_std
                src_vol = torch.clamp(src_vol + noise_src, 0, 1)
                tgt_vol = torch.clamp(tgt_vol + noise_tgt, 0, 1)
        
        return src_vol, tgt_vol


def get_augmentation(
    spatial=True,
    intensity=False,
    rotation_range=15,
    translation_range=0.1,
    scale_range=(0.9, 1.1),
    flip_prob=0.5,
    spatial_prob=0.8,
    intensity_prob=0.5,
    use_torch=True,
):
    """
    获取数据增强组合
    
    Args:
        spatial: 是否使用空间增强
        intensity: 是否使用强度增强
        rotation_range: 旋转角度范围
        translation_range: 平移范围
        scale_range: 缩放范围
        flip_prob: 翻转概率
        spatial_prob: 空间增强应用概率
        intensity_prob: 强度增强应用概率
        use_torch: 是否使用torch实现
    
    Returns:
        augmentation函数
    """
    spatial_aug = SpatialAugmentation(
        rotation_range=rotation_range,
        translation_range=translation_range,
        scale_range=scale_range,
        flip_prob=flip_prob,
        apply_prob=spatial_prob,
        use_torch=use_torch,
    ) if spatial else None
    
    intensity_aug = IntensityAugmentation(
        apply_prob=intensity_prob,
    ) if intensity else None
    
    def augment(src_vol, tgt_vol, src_seg, tgt_seg):
        # 应用空间增强
        if spatial_aug is not None:
            src_vol, tgt_vol, src_seg, tgt_seg = spatial_aug(src_vol, tgt_vol, src_seg, tgt_seg)
        
        # 应用强度增强 (仅图像)
        if intensity_aug is not None:
            src_vol, tgt_vol = intensity_aug(src_vol, tgt_vol)
        
        return src_vol, tgt_vol, src_seg, tgt_seg
    
    return augment
