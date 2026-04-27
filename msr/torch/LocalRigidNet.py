
import torch  # PyTorch主库
import torch.nn as nn  # 神经网络模块
import torch.nn.functional as F  # 常用函数（如插值、激活等）


# =========================
# 单个刚性变换预测模块
# =========================
class RigidTransformModule(nn.Module):
    def __init__(self, in_channels=32, hidden_channels=64):
        super(RigidTransformModule, self).__init__()
        
        # 3D卷积：提取局部空间特征
        self.conv1 = nn.Conv3d(in_channels, hidden_channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm3d(hidden_channels)  # 归一化，加速收敛
        self.relu = nn.ReLU(inplace=True)  # 非线性激活
        
        # 第二层卷积，进一步建模局部结构
        self.conv2 = nn.Conv3d(hidden_channels, hidden_channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm3d(hidden_channels)
        
        # 全局平均池化 -> 提取全局描述（关键：从ROI中回归刚性参数）
        self.global_pool = nn.AdaptiveAvgPool3d(1)
        
        # 全连接层：从特征回归刚性参数
        self.fc1 = nn.Linear(hidden_channels, 128)
        self.fc2 = nn.Linear(128, 64)
        
        # 输出旋转参数（rx, ry, rz）
        self.fc_rotation = nn.Linear(64, 3)
        # 输出平移参数（tx, ty, tz）
        self.fc_translation = nn.Linear(64, 3)
        
        # 初始化为0 → 初始为“单位变换”
        self.fc_rotation.weight.data.zero_()
        self.fc_rotation.bias.data.zero_()
        self.fc_translation.weight.data.zero_()
        self.fc_translation.bias.data.zero_()
    
    def forward(self, x):
        # 卷积特征提取
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.relu(self.bn2(self.conv2(x)))
        
        # 全局池化 → (B, C, 1,1,1)
        x = self.global_pool(x)
        x = x.view(x.size(0), -1)  # 展平
        
        # MLP回归
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        
        # 输出旋转和平移
        rotation = self.fc_rotation(x)
        translation = self.fc_translation(x)
        
        return rotation, translation


# =========================
# 多椎体局部刚性网络
# =========================
class LocalRigidNet(nn.Module):
    def __init__(self, feature_channels=32, num_vertebrae=5, img_size=(176, 208, 192)):
        super(LocalRigidNet, self).__init__()
        
        self.num_vertebrae = num_vertebrae  # 椎体数量
        self.img_size = img_size  # 图像尺寸
        
        # 每个椎体一个刚性预测模块（关键：局部刚性建模）
        self.rigid_modules = nn.ModuleList([
            RigidTransformModule(in_channels=feature_channels * 2)  # source+target拼接
            for _ in range(num_vertebrae)
        ])
    
    # =========================
    # 提取ROI区域特征
    # =========================
    def extract_roi_features(self, features, masks, label_id):
        batch_size = features.size(0)
        
        # 生成当前椎体mask
        mask = (masks == label_id).float()
        
        # 如果没有channel维度，则补一个
        if mask.dim() == 4:
            mask = mask.unsqueeze(1)
        
        # 将mask resize到feature尺寸
        mask = F.interpolate(mask, size=features.shape[2:], mode='nearest')
        
        # ROI特征 = feature * mask（空间裁剪）
        roi_features = features * mask
        
        return roi_features
    
    # =========================
    # 欧拉角 → 旋转矩阵
    # =========================
    def create_rotation_matrix(self, rotation_params):
        batch_size = rotation_params.size(0)
        device = rotation_params.device
        
        # 分别取出x/y/z轴旋转角
        rx, ry, rz = rotation_params[:, 0], rotation_params[:, 1], rotation_params[:, 2]
        
        zeros = torch.zeros_like(rx)
        ones = torch.ones_like(rx)
        
        # 绕x轴旋转矩阵
        Rx = torch.stack([
            torch.stack([ones, zeros, zeros], dim=1),
            torch.stack([zeros, torch.cos(rx), -torch.sin(rx)], dim=1),
            torch.stack([zeros, torch.sin(rx), torch.cos(rx)], dim=1)
        ], dim=1)
        
        # 绕y轴旋转矩阵
        Ry = torch.stack([
            torch.stack([torch.cos(ry), zeros, torch.sin(ry)], dim=1),
            torch.stack([zeros, ones, zeros], dim=1),
            torch.stack([-torch.sin(ry), zeros, torch.cos(ry)], dim=1)
        ], dim=1)
        
        # 绕z轴旋转矩阵
        Rz = torch.stack([
            torch.stack([torch.cos(rz), -torch.sin(rz), zeros], dim=1),
            torch.stack([torch.sin(rz), torch.cos(rz), zeros], dim=1),
            torch.stack([zeros, zeros, ones], dim=1)
        ], dim=1)
        
        # 最终旋转：R = Rz * Ry * Rx（标准欧拉顺序）
        R = torch.matmul(torch.matmul(Rz, Ry), Rx)
        
        return R
    
    # =========================
    # 刚性参数 → flow场
    # =========================
    def rigid_transform_to_flow(self, rotation_matrix, translation, mask, img_size):
        batch_size = rotation_matrix.size(0)
        device = rotation_matrix.device
        
        D, H, W = img_size  # 深高宽
        
        # 构建标准化坐标网格 [-1,1]
        grid_z, grid_y, grid_x = torch.meshgrid(
            torch.linspace(-1, 1, D, device=device),
            torch.linspace(-1, 1, H, device=device),
            torch.linspace(-1, 1, W, device=device),
            indexing='ij'
        )
        
        # 拼接成 (x,y,z)
        grid = torch.stack([grid_x, grid_y, grid_z], dim=-1)
        grid = grid.unsqueeze(0).repeat(batch_size, 1, 1, 1, 1)
        
        # 平移归一化（适配grid_sample坐标系）
        translation_normalized = translation.view(batch_size, 1, 1, 1, 3)
        translation_normalized = translation_normalized / torch.tensor(
            [W/2, H/2, D/2], device=device
        ).view(1, 1, 1, 1, 3)
        
        # 扩展旋转矩阵维度
        rotation_matrix_expanded = rotation_matrix.view(batch_size, 1, 1, 1, 3, 3)
        
        # 应用刚性变换
        grid_transformed = torch.matmul(
            grid.unsqueeze(-2), rotation_matrix_expanded
        ).squeeze(-2) + translation_normalized
        
        # flow = 新位置 - 原位置
        flow = grid_transformed - grid
        
        # 转为 (B, 3, D, H, W)
        flow = flow.permute(0, 4, 1, 2, 3)
        
        # mask resize
        if mask.dim() == 4:
            mask = mask.unsqueeze(1)
        mask = F.interpolate(mask, size=img_size, mode='nearest')
        
        # 只保留ROI区域的flow
        flow = flow * mask
        
        return flow
    
    # =========================
    # 前向传播
    # =========================
    def forward(self, source_features, target_features, masks):
        batch_size = source_features.size(0)
        device = source_features.device
        
        # 拼接source和target特征（用于估计变换）
        combined_features = torch.cat([source_features, target_features], dim=1)
        
        # 原图尺寸
        actual_img_size = masks.shape[1:]
        
        # 初始化总flow
        total_flow = torch.zeros(batch_size, 3, *actual_img_size, device=device)
        
        # 遍历每个椎体
        for label_id in range(1, self.num_vertebrae + 1):
            
            # 提取当前椎体ROI特征
            roi_features = self.extract_roi_features(combined_features, masks, label_id)
            
            # 如果该椎体不存在（mask为空），跳过
            if roi_features.sum() < 1e-6:
                continue
            
            # 预测刚性参数
            rotation_params, translation = self.rigid_modules[label_id - 1](roi_features)
            
            # 转换为旋转矩阵
            rotation_matrix = self.create_rotation_matrix(rotation_params)
            
            # 当前mask
            mask = (masks == label_id).float()
            
            # 生成该椎体的flow
            flow = self.rigid_transform_to_flow(
                rotation_matrix, translation, mask, actual_img_size
            )
            
            # 累加所有椎体flow
            total_flow = total_flow + flow
        
        return total_flow