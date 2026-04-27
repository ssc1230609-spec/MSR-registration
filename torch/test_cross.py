#!/usr/bin/env python

import setproctitle


import faulthandler
# 在import之后直接添加以下启用代码即可
faulthandler.enable()
# 后边正常写你的代码
import numpy as np

import os
import pdb
import random
import argparse
import time
import warnings
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.cuda.amp import autocast, GradScaler
import math
import datetime
# from torch.utils.tensorboard import SummaryWriter
from contextlib import contextmanager

# import voxelmorph with pytorch backend
os.environ['NEURITE_BACKEND'] = 'pytorch'
os.environ['VXM_BACKEND'] = 'pytorch'
import voxelmorph as vxm  # nopep8, from packages instead of source code
from voxelmorph.torch.layers import SpatialTransformer
from voxelmorph.py.utils import jacobian_determinant as jd
import sys
import pickle
import nibabel as nib


from mambamorph import generators as src_generators
from mambamorph.torch import losses as src_loss
from mambamorph.torch import networks
from mambamorph.torch import utils
from mambamorph.torch.TransMorph import CONFIGS as CONFIGS_TM
from mambamorph.torch import TransMorph as TransMorph

# parse the commandline
parser = argparse.ArgumentParser()

# data organization parameters
parser.add_argument('--subj-test', default="",
                    help='subjects used for test')
parser.add_argument('--vol-path', default="",
                    help='path to cross modality volume')
parser.add_argument('--seg-path', default="",
                    help='path to cross modality segmentation')
parser.add_argument('--load-model', required=True, help='optional model file to initialize with')
parser.add_argument('--model', type=str, default=None,
                    help='If you only load model params in load-model, you have to specify a model first')
parser.add_argument('--mode', default='mr>ct', help='register from mr -> ct')
parser.add_argument('--atlas', help='atlas filename (default: data/atlas_norm.npz)')
parser.add_argument('--multichannel', action='store_true',
                    help='specify that data has multiple channels')
parser.add_argument('--scale', type=float, default=1.0, help='scale factor of the original volume')
parser.add_argument('--chunk', action='store_true', help='whether to use chunk the volumes')

# training parameters
parser.add_argument('--gpu', default=None, help='GPU ID number(s), comma-separated (default: 0)')
parser.add_argument('--batch-size', type=int, default=1, help='batch size (default: 1)')
parser.add_argument('--cudnn-nondet', action='store_true',
                    help='disable cudnn determinism - might slow down training')

# network architecture parameters
parser.add_argument('--enc', type=int, nargs='+',
                    help='list of unet encoder filters (default: 16 32 32 32)')
parser.add_argument('--dec', type=int, nargs='+',
                    help='list of unet decorder filters (default: 32 32 32 32 32 16 16)')
parser.add_argument('--int-steps', type=int, default=7,
                    help='number of integration steps (default: 7)')
parser.add_argument('--int-downsize', type=int, default=2,
                    help='flow downsample factor for integration (default: 2)')
parser.add_argument('--bidir', action='store_true', help='enable bidirectional cost function')
parser.add_argument('--feat', action='store_true', help='whether to use feature extractor before Registraion')

# loss hyperparameters
parser.add_argument('--image-loss', default='ncc',
                    help='image reconstruction loss - can be mse or ncc (default: mse)')
parser.add_argument('--lambda', type=float, dest='weight', default=1,
                    help='weight of deformation loss (default: 0.1)')
parser.add_argument('--ignore-label', type=int, nargs='+', default=[0],
                    help='list of ignorable labels')
parser.add_argument('--cl', type=float, default=0.0, help='whether to use contrastive loss and set its weight')
parser.add_argument('--output-dir', default='./test_output', help='output directory for registered volumes and deformation fields')

# 局部刚性配准参数
parser.add_argument('--use-local-rigid', action='store_true', help='enable local rigid registration for each vertebra')
parser.add_argument('--num-vertebrae', type=int, default=4, help='number of vertebrae for local rigid registration')

args = parser.parse_args()
bidir = args.bidir

# load and prepare data
with open(args.subj_test, 'rb') as file:
    test_subject = pickle.load(file)

data_example = vxm.py.utils.load_volfile(
    args.seg_path + test_subject[0] + '.nii.gz')
inshape = tuple([int(old_size * args.scale) for old_size in data_example.shape])
labels_in = np.unique(data_example)

# no need to append an extra feature axis if data is multichannel
add_feat_axis = not args.multichannel

# device handling
if args.gpu:
    gpus = args.gpu.split(',')
    nb_gpus = len(gpus)
    device = 'cuda'
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    assert np.mod(args.batch_size, nb_gpus) == 0, \
        'Batch size (%d) should be a multiple of the nr of gpus (%d)' % (args.batch_size, nb_gpus)
    # enabling cudnn determinism appears to speed up training by a lot
    torch.backends.cudnn.deterministic = not args.cudnn_nondet
else:
    nb_gpus = 0
    device = 'cpu'

torch.cuda.reset_max_memory_allocated()
torch.cuda.reset_max_memory_cached()

# Prepare a model
if args.model is None:
    model = torch.load(args.load_model)
elif args.model == 'vm':
    model = networks.VxmDense.load(args.load_model, device)
elif args.model == 'mm-feat':
    config = CONFIGS_TM['MambaMorph']
    config.img_size = inshape  # 使用实际数据尺寸
    config.use_local_rigid = args.use_local_rigid
    config.num_vertebrae = args.num_vertebrae
    model = TransMorph.MambaMorphFeat(config)
    model.load_state_dict(torch.load(args.load_model))
elif args.model == 'mm-swin-feat':
    # MambaMorphSwinFeat: 带特征提取器的Mamba-Swin融合模型
    config = CONFIGS_TM['MambaMorphSwin']
    config.img_size = inshape
    config.int_steps = args.int_steps
    config.int_downsize = args.int_downsize
    # 局部刚性配准配置
    config.use_local_rigid = args.use_local_rigid
    config.num_vertebrae = args.num_vertebrae
    model = TransMorph.MambaMorphSwinFeat(config)
    model.load_state_dict(torch.load(args.load_model))
    print(f"MambaMorphSwinFeat: Fusion of Mamba and SwinTransformer with feature extractor")

if nb_gpus > 1:
    # use multiple GPUs via DataParallel
    model = torch.nn.DataParallel(model)

# prepare the model for training and send to device
model.to(device)
total_params = sum([param.numel() for param in model.parameters()])
# set the ignorable labels
if len(args.ignore_label) > 0:
    label_ignore = ~np.isin(labels_in, args.ignore_label)
else:
    label_ignore = np.ones(len(labels_in)).bool()
label_ignore = torch.from_numpy(label_ignore).to(device)

# test loops
if test_subject is not None:
    np.random.seed(0)
    first_mod = args.mode.split('>')[0]
    second_mod = args.mode.split('>')[1]

    test_vol_moving = [os.path.join(args.vol_path, item + f"_{first_mod}.nii.gz") for item in test_subject]
    test_vol_fixed = [os.path.join(args.vol_path, item + f"_{second_mod}.nii.gz") for item in test_subject]
    test_seg = [os.path.join(args.seg_path, item + f".nii.gz") for item in test_subject]
    model.eval()
    transform_model = SpatialTransformer(inshape, mode='nearest')  # STN
    transform_model.to(device)
    
    # Create output directories
    os.makedirs(args.output_dir, exist_ok=True)
    warped_dir = os.path.join(args.output_dir, 'warped_volumes')
    flow_dir = os.path.join(args.output_dir, 'deformation_fields')
    warped_seg_dir = os.path.join(args.output_dir, 'warped_labels')
    os.makedirs(warped_dir, exist_ok=True)
    os.makedirs(flow_dir, exist_ok=True)
    os.makedirs(warped_seg_dir, exist_ok=True)
    

    dice_means = []
    anatomical_dice = []
    Jacobian_ratio = []
    HD95 = []
    infer_time = []
    rec = 0
    with torch.no_grad():
        for idx in range(len(test_subject)):
            rec += 1
            print(f"Processing {rec}/{len(test_subject)}: {test_subject[idx]}", end='')
            # read the images and labels from files (same subject, different modalities)
            moving, moving_affine = vxm.py.utils.load_volfile(test_vol_moving[idx], add_batch_axis=True, np_var='vol',
                                               add_feat_axis=add_feat_axis, resize_factor=args.scale, ret_affine=True)
            moving = utils.minmax_norm(moving)
            moving_seg = vxm.py.utils.load_volfile(test_seg[idx], add_batch_axis=True,
                                                   np_var='seg', add_feat_axis=add_feat_axis, resize_factor=args.scale)

            fixed, fixed_affine = vxm.py.utils.load_volfile(test_vol_fixed[idx], add_batch_axis=True, np_var='vol',
                                              add_feat_axis=add_feat_axis, resize_factor=args.scale, ret_affine=True)
            fixed = utils.minmax_norm(fixed)
            fixed_seg = vxm.py.utils.load_volfile(test_seg[idx], add_batch_axis=True,
                                                  np_var='seg', add_feat_axis=add_feat_axis, resize_factor=args.scale)

            moving_seg = src_generators.split_seg_global(moving_seg, labels_in)
            fixed_seg = src_generators.split_seg_global(fixed_seg, labels_in)

            # Infer
            input_moving = torch.from_numpy(moving).to(device).float().permute(0, 4, 1, 2, 3)
            input_fixed = torch.from_numpy(fixed).to(device).float().permute(0, 4, 1, 2, 3)
            
            # 准备椎骨掩码（如果使用局部刚性配准）
            if args.use_local_rigid:
                input_seg = torch.from_numpy(moving_seg).to(device).float().permute(0, 4, 1, 2, 3)
                src_mask = torch.argmax(input_seg, dim=1, keepdim=False)
            else:
                src_mask = None
            
            # predict
            start_time = time.time()
            if args.use_local_rigid:
                ret_dict = model(input_moving, input_fixed, source_mask=src_mask, return_pos_flow=True)
            else:
                ret_dict = model(input_moving, input_fixed, return_pos_flow=True)
            end_time = time.time()
            duration = end_time - start_time
            if idx == 0:
                memory_allocated = torch.cuda.max_memory_allocated()
                memory_cached = torch.cuda.max_memory_reserved()
                Gb_consumed = memory_cached / 1e9
            moved = ret_dict['moved_vol']
            # 优先使用pos_flow（包含局部刚性变形），否则使用preint_flow
            warp = ret_dict.get('pos_flow', ret_dict['preint_flow'])

            # Warp the moving segment
            input_seg = torch.from_numpy(moving_seg).to(device).float().permute(0, 4, 1, 2, 3)
            warped_seg = transform_model(input_seg, warp).squeeze()

            # Dice
            overlap = vxm.py.utils.dice(np.argmax(warped_seg.cpu().numpy(), axis=0),
                                        np.argmax(fixed_seg, axis=-1).squeeze(), include_zero=True)
            current_dice = np.mean(overlap[label_ignore.cpu().numpy()] * 100)
            dice_means.append(current_dice)
            anatomical_dice.append(overlap[label_ignore.cpu().numpy()])

            # |J|<0
            minus_ratio = utils.negative_jacobin(warp[0].permute(1, 2, 3, 0).cpu().numpy())
            Jacobian_ratio.append(minus_ratio)
            
            # Print current sample results
            print(f"Dice: {current_dice:.2f}%, |J|<0: {minus_ratio*100:.4f}%")

            # 95% Hausdorff distance
            warped_seg_np = warped_seg.cpu().permute(1, 2, 3, 0).numpy()
            fixed_seg_np = fixed_seg.squeeze()
            HD95_per_label = []
            for label_idx in np.where(~np.isin(labels_in, args.ignore_label))[0]:
                HD95_each = utils.hausdorff_distance(warped_seg_np[..., label_idx],
                                                     fixed_seg_np[..., label_idx], percentage=95)
                HD95_per_label.append(HD95_each)
            HD95.append(np.mean(HD95_per_label))
            # Calculating inference time
            if idx > 0:
                infer_time.append(duration)
            
            # Save warped volume and deformation field
            subject_name = test_subject[idx]
            
            # Save warped volume (use fixed image's affine to preserve spacing)
            warped_vol = moved[0, 0].cpu().numpy()  # Remove batch and channel dimensions
            warped_img = nib.Nifti1Image(warped_vol, affine=fixed_affine)
            warped_filename = os.path.join(warped_dir, f'{subject_name}_{first_mod}_to_{second_mod}_warped.nii.gz')
            nib.save(warped_img, warped_filename)
            
            # Save deformation field (flow field) with fixed image's affine
            flow_field = warp[0].permute(1, 2, 3, 0).cpu().numpy()  # [H, W, D, 3]
            flow_img = nib.Nifti1Image(flow_field, affine=fixed_affine)
            flow_filename = os.path.join(flow_dir, f'{subject_name}_{first_mod}_to_{second_mod}_flow.nii.gz')
            nib.save(flow_img, flow_filename)
            
            # Save warped label (convert from one-hot to label indices)
            warped_seg_label = np.argmax(warped_seg.cpu().numpy(), axis=0).astype(np.uint8)
            warped_seg_img = nib.Nifti1Image(warped_seg_label, affine=fixed_affine)
            warped_seg_filename = os.path.join(warped_seg_dir, f'{subject_name}_{first_mod}_to_{second_mod}_warped_label.nii.gz')
            nib.save(warped_seg_img, warped_seg_filename)
            
            print(f'Saved: {warped_filename}')
            print(f'Saved: {flow_filename}')
            print(f'Saved: {warped_seg_filename}')

    anatomical_dice = np.stack(anatomical_dice)
    anatomical_dice_mean = anatomical_dice.mean(axis=0)
    anatomical_dice_std = anatomical_dice.std(axis=0)


    # Format results
    results_text = []
    results_text.append('=' * 60)
    results_text.append('EVALUATION RESULTS')
    results_text.append('=' * 60)
    results_text.append(f'Average Dice:              {np.mean(dice_means):.2f} ± {np.std(dice_means):.2f}%')
    results_text.append(f'|J| < 0 percentage:        {np.mean(Jacobian_ratio)*100:.4f} ± {np.std(Jacobian_ratio)*100:.4f}%')
    results_text.append(f'HD95:                      {np.mean(HD95):.2f} ± {np.std(HD95):.2f} mm')
    results_text.append(f'Inference time:            {np.mean(infer_time):.4f} ± {np.std(infer_time):.4f} s')
    results_text.append(f'Memory occupation:         {Gb_consumed:.4f} GB')
    results_text.append(f'Model parameters:          {total_params / 1e6:.2f} M')
    results_text.append(f'Anatomical Dice (mean):    {anatomical_dice_mean}')
    results_text.append(f'Anatomical Dice (std):     {anatomical_dice_std}')
    results_text.append('=' * 60)
    
    # Print to console
    print()
    for line in results_text:
        print(line)
    
    # Save to file
    results_file = os.path.join(args.output_dir, 'test_results.txt')
    with open(results_file, 'w') as f:
        f.write('\n'.join(results_text))
    
    print(f'Results saved to: {results_file}')
    print('Testing completed!')
