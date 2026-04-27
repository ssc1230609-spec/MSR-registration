import pdb
# import voxelmorph as vxm
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math

cl_cfg = dict(
    pre_select_pos_number=10000,  # default 2000
    after_select_pos_number=100,  # default 100
    pre_select_neg_number=1000,  # default 2000
    after_select_neg_number=250,  # default 500
    positive_distance=0.95,  # default 2.
    ignore_distance=40.,
    coarse_positive_distance=25.,
    coarse_ignore_distance=5.,
    coarse_z_thres=6.,
    coarse_pre_select_neg_number=250,
    coarse_after_select_neg_number=200,
    fine_temperature=0.25,  # default 0.5
    coarse_temperature=0.5,
    select_pos_num=1000,
    select_neg_num=5000, )


class NCC:
    """
    Local (over window) normalized cross correlation loss.
    """

    def __init__(self, win=None):
        self.win = [win]*3

    def loss(self, y_true, y_pred, weight=None, return_per_loss=False,ignore_label=None):

        Ii = y_true
        Ji = y_pred


        # get dimension of volume
        # assumes Ii, Ji are sized [batch_size, *vol_shape, nb_feats]
        ndims = len(list(Ii.size())) - 2
        assert ndims in [1, 2, 3], "volumes should be 1 to 3 dimensions. found: %d" % ndims

        # set window size
        win = [5] * ndims if self.win is None else self.win

        # compute filters
        sum_filt = torch.ones([1, 1, *win]).to("cuda")

        pad_no = math.floor(win[0] / 2)

        if ndims == 1:
            stride = (1)
            padding = (pad_no)
        elif ndims == 2:
            stride = (1, 1)
            padding = (pad_no, pad_no)
        else:
            stride = (1, 1, 1)
            padding = (pad_no, pad_no, pad_no)

        # get convolution function
        conv_fn = getattr(F, 'conv%dd' % ndims)

        # compute CC squares
        I2 = Ii * Ii
        J2 = Ji * Ji
        IJ = Ii * Ji

        I_sum = conv_fn(Ii, sum_filt, stride=stride, padding=padding)
        J_sum = conv_fn(Ji, sum_filt, stride=stride, padding=padding)
        I2_sum = conv_fn(I2, sum_filt, stride=stride, padding=padding)
        J2_sum = conv_fn(J2, sum_filt, stride=stride, padding=padding)
        IJ_sum = conv_fn(IJ, sum_filt, stride=stride, padding=padding)

        win_size = np.prod(win)
        u_I = I_sum / win_size
        u_J = J_sum / win_size

        cross = IJ_sum - u_J * I_sum - u_I * J_sum + u_I * u_J * win_size
        I_var = I2_sum - 2 * u_I * I_sum + u_I * u_I * win_size
        J_var = J2_sum - 2 * u_J * J_sum + u_J * u_J * win_size

        cc = cross * cross / (I_var * J_var + 1e-5)
        if weight is not None:
            B = len(cc)
            assert len(weight) == B, "The length of data weights must be equal to the batch value."
            assert 0.99 < weight.sum().item() < 1.1, "The weights of data must sum to 1."
            weighted_loss = torch.tensor(0., device=cc.device)
            per_loss = torch.zeros([B], dtype=torch.float32, device=cc.device)
            for idx in range(B):
                item_loss = -torch.mean(cc[idx])
                weighted_loss += item_loss * weight[idx]
                per_loss[idx] = item_loss
            if return_per_loss:
                return weighted_loss, per_loss
            else:
                return weighted_loss
        else:
            return -torch.mean(cc)


class MSE:
    """
    Mean squared error loss.
    """

    def loss(self, y_true, y_pred, weight=None, return_per_loss=False):
        if weight is not None:
            B = len(y_true)
            assert len(weight) == B, "The length of data weights must be equal to the batch value."
            assert 0.99 < weight.sum().item() < 1.1, "The weights of data must sum to 1."
            weighted_loss = torch.tensor(0., device=y_true.device)
            per_loss = torch.zeros([B], dtype=torch.float32, device=y_true.device)
            for idx in range(B):
                item_loss = torch.mean((y_true[idx] - y_pred[idx]) ** 2)
                weighted_loss += item_loss * weight[idx]
                per_loss[idx] = item_loss
            if return_per_loss:
                return weighted_loss, per_loss
            else:
                return weighted_loss
        else:
            return torch.mean((y_true - y_pred) ** 2)


class Dice:
    """
    N-D dice for segmentation
    """

    def loss(self, y_true, y_pred, weight=None, return_per_loss=False, ignore_label=None):
        ndims = len(list(y_pred.size())) - 2
        vol_axes = list(range(2, ndims + 2))#在哪个维度上进行dice的计算
        if weight is not None:#是否给样本计算dice时进行加权
            B = len(y_true)
            assert len(weight) == B, "The length of data weights must be equal to the batch value."
            assert 0.99 < weight.sum().item() < 1.1, "The weights of data must sum to 1."
            weighted_loss = torch.tensor(0., device=y_true.device)
            per_loss = torch.zeros([B], dtype=torch.float32, device=y_true.device)
            for idx in range(B):
                top = 2 * (y_true[idx:idx + 1] * y_pred[idx:idx + 1]).sum(dim=vol_axes)
                bottom = torch.clamp((y_true[idx:idx + 1] + y_pred[idx:idx + 1]).sum(dim=vol_axes), min=1e-5)
                if ignore_label is not None:
                    item_dice = -torch.mean(top[:, ignore_label] / bottom[:, ignore_label])
                else:
                    item_dice = -torch.mean(top / bottom)
                weighted_loss += item_dice * weight[idx]
                per_loss[idx] = item_dice
            if return_per_loss:
                return weighted_loss, per_loss
            else:
                return weighted_loss
        else:
            top = 2 * (y_true * y_pred).sum(dim=vol_axes)#计算dice的分子
            bottom = torch.clamp((y_true + y_pred).sum(dim=vol_axes), min=1e-5)#计算dice的分母
            if ignore_label is not None:
                dice = torch.mean(top[:, ignore_label] / bottom[:, ignore_label])
            else:
                dice = torch.mean(top / bottom)
        return -dice

    def each_dice(self, y_true, y_pred, ignore_label=None):
        ndims = len(list(y_pred.size())) - 2
        vol_axes = list(range(2, ndims + 2))
        top = 2 * (y_true * y_pred).sum(dim=vol_axes)
        bottom = torch.clamp((y_true + y_pred).sum(dim=vol_axes), min=1e-5)
        if ignore_label is not None:
            dice = top[:, ignore_label] / bottom[:, ignore_label]
        else:
            dice = top / bottom
        return dice


class Grad:
    """
    N-D gradient loss.
    """

    def __init__(self, penalty='l1', loss_mult=None):
        self.penalty = penalty
        self.loss_mult = loss_mult

    def _diffs(self, y):
        vol_shape = [n for n in y.shape][2:]
        ndims = len(vol_shape)

        df = [None] * ndims
        for i in range(ndims):
            d = i + 2
            # permute dimensions
            r = [d, *range(0, d), *range(d + 1, ndims + 2)]
            y = y.permute(r)
            dfi = y[1:, ...] - y[:-1, ...]

            # permute back
            # note: this might not be necessary for this loss specifically,
            # since the results are just summed over anyway.
            r = [*range(d - 1, d + 1), *reversed(range(1, d - 1)), 0, *range(d + 1, ndims + 2)]
            df[i] = dfi.permute(r)

        return df

    def loss(self, _, y_pred, weight=None, return_per_loss=False, ignore_label=None):
        if self.penalty == 'l1':
            dif = [torch.abs(f) for f in self._diffs(y_pred)]
        else:
            assert self.penalty == 'l2', 'penalty can only be l1 or l2. Got: %s' % self.penalty
            dif = [f * f for f in self._diffs(y_pred)]
        df = [torch.mean(torch.flatten(f, start_dim=1), dim=-1) for f in dif]
        grad = sum(df) / len(df)

        if self.loss_mult is not None:
            grad *= self.loss_mult

        if weight is not None:
            B = len(grad)
            assert len(weight) == B, "The length of data weights must be equal to the batch value."
            assert 0.99 < weight.sum().item() < 1.1, "The weights of data must sum to 1."
            weighted_loss = torch.tensor(0., device=grad.device)
            per_loss = torch.zeros([B], dtype=torch.float32, device=grad.device)
            for idx in range(B):
                weighted_loss += grad[idx] * weight[idx]
                per_loss[idx] = grad[idx]
            if return_per_loss:
                return weighted_loss, per_loss
            else:
                return weighted_loss
        else:
            return grad.mean()


class MIND_loss(torch.nn.Module):
    """
    MIND-SSC (Modality Independent Neighbourhood Descriptor) loss for multi-modal image registration.
    Reference: http://mpheinrich.de/pub/miccai2013_943_mheinrich.pdf
    """

    def __init__(self, win=None, radius=2, dilation=2):
        super(MIND_loss, self).__init__()
        self.win = win
        self.radius = radius
        self.dilation = dilation
        
        # Pre-compute fixed kernels
        kernel_size = radius * 2 + 1
        
        # define start and end locations for self-similarity pattern
        six_neighbourhood = torch.Tensor([[0, 1, 1],
                                          [1, 1, 0],
                                          [1, 0, 1],
                                          [1, 1, 2],
                                          [2, 1, 1],
                                          [1, 2, 1]]).long()
        
        # squared distances
        dist = self.pdist_squared(six_neighbourhood.t().unsqueeze(0)).squeeze(0)
        
        # define comparison mask
        x, y = torch.meshgrid(torch.arange(6), torch.arange(6), indexing='ij')
        mask = ((x > y).view(-1) & (dist == 2).view(-1))
        
        # build kernel
        idx_shift1 = six_neighbourhood.unsqueeze(1).repeat(1, 6, 1).view(-1, 3)[mask, :]
        idx_shift2 = six_neighbourhood.unsqueeze(0).repeat(6, 1, 1).view(-1, 3)[mask, :]
        
        # Store as buffers (will be moved to device automatically)
        mshift1 = torch.zeros(12, 1, 3, 3, 3)
        mshift1.view(-1)[torch.arange(12) * 27 + idx_shift1[:, 0] * 9 + idx_shift1[:, 1] * 3 + idx_shift1[:, 2]] = 1
        mshift2 = torch.zeros(12, 1, 3, 3, 3)
        mshift2.view(-1)[torch.arange(12) * 27 + idx_shift2[:, 0] * 9 + idx_shift2[:, 1] * 3 + idx_shift2[:, 2]] = 1
        
        self.register_buffer('mshift1', mshift1)
        self.register_buffer('mshift2', mshift2)
        
        # Register permutation indices as buffer to ensure it's on the same device
        perm_indices = torch.tensor([6, 8, 1, 11, 2, 10, 0, 7, 9, 4, 5, 3], dtype=torch.long)
        self.register_buffer('perm_indices', perm_indices)

    def pdist_squared(self, x):
        xx = (x ** 2).sum(dim=1).unsqueeze(2)
        yy = xx.permute(0, 2, 1)
        dist = xx + yy - 2.0 * torch.bmm(x.permute(0, 2, 1), x)
        dist[dist != dist] = 0
        dist = torch.clamp(dist, 0.0, np.inf)
        return dist

    def MINDSSC(self, img):
        # see http://mpheinrich.de/pub/miccai2013_943_mheinrich.pdf for details on the MIND-SSC descriptor
        
        # kernel size
        kernel_size = self.radius * 2 + 1
        
        # Use pre-computed kernels
        rpad1 = nn.ReplicationPad3d(self.dilation)
        rpad2 = nn.ReplicationPad3d(self.radius)

        # compute patch-ssd
        ssd = F.avg_pool3d(rpad2(
            (F.conv3d(rpad1(img), self.mshift1, dilation=self.dilation) - F.conv3d(rpad1(img), self.mshift2, dilation=self.dilation)) ** 2),
                           kernel_size, stride=1)

        # MIND equation
        mind = ssd - torch.min(ssd, 1, keepdim=True)[0]
        mind_var = torch.mean(mind, 1, keepdim=True)
        # Use fixed epsilon to avoid division by zero, remove dynamic .item() calls
        mind = mind / (mind_var + 1e-5)
        mind = torch.exp(-mind)

        # permute to have same ordering as C++ code
        mind = mind[:, self.perm_indices, :, :, :]

        return mind

    def loss(self, y_pred, y_true):
        return torch.mean((self.MINDSSC(y_pred) - self.MINDSSC(y_true)) ** 2)


class MutualInformation:
    """
    Mutual Information loss for multi-modal image registration.
    Uses Parzen window (Gaussian kernel) density estimation for histogram computation.
    """
    
    def __init__(self, num_bins=32, sigma_ratio=1.0, minval=0., maxval=1., normalize=True):
        """
        Args:
            num_bins: Number of bins for histogram (default: 32)
            sigma_ratio: Ratio to scale the Gaussian sigma (default: 1.0)
            minval: Minimum intensity value for binning (default: 0.)
            maxval: Maximum intensity value for binning (default: 1.)
            normalize: Whether to normalize MI by entropy (default: True)
        """
        self.num_bins = num_bins
        self.sigma_ratio = sigma_ratio
        self.minval = minval
        self.maxval = maxval
        self.normalize = normalize
        
        # Compute bin centers
        self.bin_centers = np.linspace(minval, maxval, num=num_bins)
        
        # Compute sigma for Gaussian approximation
        self.sigma = np.mean(np.diff(self.bin_centers)) * sigma_ratio
        self.preterm = 1.0 / (2.0 * self.sigma ** 2)
    
    def _compute_mi(self, y_true, y_pred):
        """
        Compute mutual information between two images.
        
        Args:
            y_true: Ground truth image [B, C, ...]
            y_pred: Predicted/warped image [B, C, ...]
            
        Returns:
            Mutual information value (scalar)
        """
        device = y_true.device
        
        # Clamp intensities to valid range
        y_true = torch.clamp(y_true, self.minval, self.maxval)
        y_pred = torch.clamp(y_pred, self.minval, self.maxval)
        
        # Flatten spatial dimensions: [B, C, ...] -> [B, N]
        y_true_flat = y_true.reshape(y_true.shape[0], -1)
        y_pred_flat = y_pred.reshape(y_pred.shape[0], -1)
        
        # Add dimension for broadcasting: [B, N] -> [B, N, 1]
        y_true_flat = y_true_flat.unsqueeze(-1)
        y_pred_flat = y_pred_flat.unsqueeze(-1)
        
        # Create bin centers tensor: [num_bins]
        bin_centers = torch.tensor(self.bin_centers, dtype=torch.float32, device=device)
        
        # Reshape bin centers for broadcasting: [1, 1, num_bins]
        bin_centers = bin_centers.view(1, 1, -1)
        
        # Compute Gaussian approximation of histograms
        # I_a, I_b: [B, N, num_bins]
        I_a = torch.exp(-self.preterm * (y_true_flat - bin_centers) ** 2)
        I_a = I_a / (torch.sum(I_a, dim=-1, keepdim=True) + 1e-10)
        
        I_b = torch.exp(-self.preterm * (y_pred_flat - bin_centers) ** 2)
        I_b = I_b / (torch.sum(I_b, dim=-1, keepdim=True) + 1e-10)
        
        # Compute joint probability: [B, num_bins, num_bins]
        pab = torch.bmm(I_a.transpose(1, 2), I_b)
        pab = pab / y_true_flat.shape[1]  # Normalize by number of voxels
        
        # Compute marginal probabilities: [B, 1, num_bins]
        pa = torch.mean(I_a, dim=1, keepdim=True)
        pb = torch.mean(I_b, dim=1, keepdim=True)
        
        # Compute product of marginals: [B, num_bins, num_bins]
        papb = torch.bmm(pa.transpose(1, 2), pb) + 1e-10
        
        # Compute mutual information: MI = sum(pab * log(pab / papb))
        # Add small epsilon to avoid log(0)
        mi = torch.sum(pab * torch.log((pab + 1e-10) / papb), dim=[1, 2])
        
        # Optional: Normalize MI by joint entropy for numerical stability
        if self.normalize:
            # Normalized MI: NMI = MI / H(A,B) where H(A,B) is joint entropy
            h_ab = -torch.sum(pab * torch.log(pab + 1e-10), dim=[1, 2])
            mi = mi / (h_ab + 1e-10)
        
        return mi.mean()  # Average across batch
    
    def loss(self, y_true, y_pred, weight=None, return_per_loss=False, ignore_label=None):
        """
        Compute MI loss (negative mutual information for minimization).
        
        Args:
            y_true: Ground truth image
            y_pred: Predicted/warped image
            weight: Optional sample weights
            return_per_loss: Return per-sample losses
            ignore_label: Placeholder for compatibility
            
        Returns:
            Negative mutual information (to minimize)
        """
        if weight is not None:
            B = len(y_true)
            assert len(weight) == B, "The length of data weights must be equal to the batch value."
            assert 0.99 < weight.sum().item() < 1.1, "The weights of data must sum to 1."
            
            device = y_true.device
            weighted_loss = torch.tensor(0., device=device)
            per_loss = torch.zeros([B], dtype=torch.float32, device=device)
            
            for idx in range(B):
                # Compute MI for single sample
                single_mi = self._compute_mi(y_true[idx:idx+1], y_pred[idx:idx+1])
                item_loss = -single_mi
                weighted_loss += item_loss * weight[idx]
                per_loss[idx] = item_loss
            
            if return_per_loss:
                return weighted_loss, per_loss
            else:
                return weighted_loss
        else:
            # Compute MI for entire batch
            mi = self._compute_mi(y_true, y_pred)
            return -mi  # Return negative MI for minimization


class LocalMutualInformation:
    """
    Local Mutual Information loss using sliding windows.
    Computes MI in local neighborhoods for better spatial adaptation.
    """
    
    def __init__(self, num_bins=32, sigma_ratio=1.0, minval=0., maxval=1., win=5):
        """
        Args:
            num_bins: Number of bins for histogram (default: 32)
            sigma_ratio: Ratio to scale the Gaussian sigma (default: 1.0)
            minval: Minimum intensity value for binning (default: 0.)
            maxval: Maximum intensity value for binning (default: 1.)
            win: Window size for local computation (default: 5)
        """
        self.num_bins = num_bins
        self.sigma_ratio = sigma_ratio
        self.minval = minval
        self.maxval = maxval
        self.win = win
        
        # Compute bin centers
        self.bin_centers = np.linspace(minval, maxval, num=num_bins)
        
        # Compute sigma for Gaussian approximation
        self.sigma = np.mean(np.diff(self.bin_centers)) * sigma_ratio
        self.preterm = 1.0 / (2.0 * self.sigma ** 2)
    
    def _compute_local_mi(self, y_true, y_pred):
        """
        Compute local mutual information using sliding windows.
        
        Args:
            y_true: Ground truth image [B, C, H, W, D]
            y_pred: Predicted/warped image [B, C, H, W, D]
            
        Returns:
            Local MI averaged across all windows
        """
        device = y_true.device
        ndims = len(y_true.shape) - 2
        
        # Clamp intensities
        y_true = torch.clamp(y_true, self.minval, self.maxval)
        y_pred = torch.clamp(y_pred, self.minval, self.maxval)
        
        # Create unfold operation for local patches
        if ndims == 3:
            # For 3D volumes, use custom sliding window
            B, C, H, W, D = y_true.shape
            pad = self.win // 2
            
            # Pad input
            y_true_pad = F.pad(y_true, [pad]*6, mode='replicate')
            y_pred_pad = F.pad(y_pred, [pad]*6, mode='replicate')
            
            # Collect local patches (strided for efficiency)
            stride = max(1, self.win // 2)
            patches_true = []
            patches_pred = []
            
            for h in range(0, H, stride):
                for w in range(0, W, stride):
                    for d in range(0, D, stride):
                        patch_true = y_true_pad[:, :, h:h+self.win, w:w+self.win, d:d+self.win]
                        patch_pred = y_pred_pad[:, :, h:h+self.win, w:w+self.win, d:d+self.win]
                        patches_true.append(patch_true.reshape(B, -1))
                        patches_pred.append(patch_pred.reshape(B, -1))
            
            # Stack patches: [B, num_patches, patch_size]
            patches_true = torch.stack(patches_true, dim=1)
            patches_pred = torch.stack(patches_pred, dim=1)
        else:
            raise NotImplementedError("Local MI currently only supports 3D volumes")
        
        # Compute MI for each patch
        bin_centers = torch.tensor(self.bin_centers, dtype=torch.float32, device=device)
        bin_centers = bin_centers.view(1, 1, 1, -1)  # [1, 1, 1, num_bins]
        
        # Add dimension: [B, num_patches, patch_size, 1]
        patches_true = patches_true.unsqueeze(-1)
        patches_pred = patches_pred.unsqueeze(-1)
        
        # Compute histograms: [B, num_patches, patch_size, num_bins]
        I_a = torch.exp(-self.preterm * (patches_true - bin_centers) ** 2)
        I_a = I_a / (torch.sum(I_a, dim=-1, keepdim=True) + 1e-10)
        
        I_b = torch.exp(-self.preterm * (patches_pred - bin_centers) ** 2)
        I_b = I_b / (torch.sum(I_b, dim=-1, keepdim=True) + 1e-10)
        
        # Marginal probabilities: [B, num_patches, 1, num_bins]
        pa = torch.mean(I_a, dim=2, keepdim=True)
        pb = torch.mean(I_b, dim=2, keepdim=True)
        
        # Joint probability for each patch
        # Reshape for batch matrix multiplication
        B, num_patches, patch_size, num_bins = I_a.shape
        I_a_reshaped = I_a.reshape(B * num_patches, patch_size, num_bins)
        I_b_reshaped = I_b.reshape(B * num_patches, patch_size, num_bins)
        
        pab = torch.bmm(I_a_reshaped.transpose(1, 2), I_b_reshaped)
        pab = pab.reshape(B, num_patches, num_bins, num_bins)
        pab = pab / patch_size
        
        # Product of marginals
        papb = torch.matmul(pa.transpose(-1, -2), pb) + 1e-10
        
        # Compute MI for each patch
        mi = torch.sum(pab * torch.log((pab + 1e-10) / papb), dim=[2, 3])
        
        # Average MI across all patches and batch
        return mi.mean()
    
    def loss(self, y_true, y_pred, weight=None, return_per_loss=False, ignore_label=None):
        """
        Compute local MI loss.
        
        Args:
            y_true: Ground truth image
            y_pred: Predicted/warped image
            weight: Optional sample weights
            return_per_loss: Return per-sample losses
            ignore_label: Placeholder for compatibility
            
        Returns:
            Negative local mutual information
        """
        if weight is not None:
            B = len(y_true)
            assert len(weight) == B, "The length of data weights must be equal to the batch value."
            assert 0.99 < weight.sum().item() < 1.1, "The weights of data must sum to 1."
            
            device = y_true.device
            weighted_loss = torch.tensor(0., device=device)
            per_loss = torch.zeros([B], dtype=torch.float32, device=device)
            
            for idx in range(B):
                single_mi = self._compute_local_mi(y_true[idx:idx+1], y_pred[idx:idx+1])
                item_loss = -single_mi
                weighted_loss += item_loss * weight[idx]
                per_loss[idx] = item_loss
            
            if return_per_loss:
                return weighted_loss, per_loss
            else:
                return weighted_loss
        else:
            mi = self._compute_local_mi(y_true, y_pred)
            return -mi
