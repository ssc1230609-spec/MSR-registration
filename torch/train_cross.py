#!/usr/bin/env python

"""
Example script to train a VoxelMorph model.

You will likely have to customize this script slightly to accommodate your own data. All images
should be appropriately cropped and scaled to values between 0 and 1.

If an atlas file is provided with the --atlas flag, then scan-to-atlas training is performed.
Otherwise, registration will be scan-to-scan.

If you use this code, please cite the following, and read function docs for further info/citations.

    VoxelMorph: A Learning Framework for Deformable Medical Image Registration G. Balakrishnan, A.
    Zhao, M. R. Sabuncu, J. Guttag, A.V. Dalca. IEEE TMI: Transactions on Medical Imaging. 38(8). pp
    1788-1800. 2019. 

    or

    Unsupervised Learning for Probabilistic Diffeomorphic Registration for Images and Surfaces
    A.V. Dalca, G. Balakrishnan, J. Guttag, M.R. Sabuncu. 
    MedIA: Medical Image Analysis. (57). pp 226-236, 2019 

Copyright 2020 Adrian V. Dalca

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in
compliance with the License. You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is
distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or
implied. See the License for the specific language governing permissions and limitations under the
License.
"""
import setproctitle
import numpy as np
import os
import pdb
import random
import argparse
import time
import warnings
import matplotlib

# CRITICAL: Set GPU before importing torch/CUDA libraries
import sys
for i, arg in enumerate(sys.argv):
    if arg == '--gpu' and i + 1 < len(sys.argv):
        os.environ['CUDA_VISIBLE_DEVICES'] = sys.argv[i + 1]
        break

matplotlib.use('Agg')
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
import sys
import pickle
from mambamorph import generators as src_generators
from mambamorph.torch import losses as src_loss
from mambamorph.torch import networks
from mambamorph.torch import utils
from mambamorph.torch.TransMorph import CONFIGS as CONFIGS_TM
from mambamorph.torch import TransMorph as TransMorph
from mambamorph.augmentation import get_augmentation

# parse the commandline
parser = argparse.ArgumentParser()

# data organization parameters
parser.add_argument('--subj-train', default="",
                    help='subjects used for train')
parser.add_argument('--subj-val', default="",  
                    help='subjects used for validation')
parser.add_argument('--subj-test', default="",
                    help='subjects used for test')
parser.add_argument('--vol-path', default="",
                    help='path to cross modality volume')
parser.add_argument('--seg-path', default="",
                    help='path to cross modality segmentation')
parser.add_argument('--same_subject',action='store_true',help='spine should be in the same subject')
parser.add_argument('--model-dir', default='models',
                    help='model output directory (default: models)')
parser.add_argument('--mode', default='mr>ct', help='register from mr -> ct')
parser.add_argument('--atlas', help='atlas filename (default: data/atlas_norm.npz)')
parser.add_argument('--multichannel', action='store_true',
                    help='specify that data has multiple channels')#CT和MRI是单通道
parser.add_argument('--scale', type=float, default=1.0, help='scale factor of the original volume')
parser.add_argument('--chunk', action='store_true', help='whether to use chunk the volumes')#是否进行分块处理

# training parameters
parser.add_argument('--gpu', default=None, help='GPU ID number(s), comma-separated (default: 0)')
parser.add_argument('--batch-size', type=int, default=1, help='batch size (default: 1)')
parser.add_argument('--epochs', type=int, default=1500,
                    help='number of training epochs (default: 1500)')
parser.add_argument('--steps-per-epoch', type=int, default=1000,
                    help='frequency of model saves (default: 100)')#模型保存频率
parser.add_argument('--load-model', help='optional model file to initialize with')
parser.add_argument('--load-model-dds', help='optional model file to initialize with')
parser.add_argument('--initial-epoch', type=int, default=0,
                    help='initial epoch number (default: 0)')
parser.add_argument('--lr', type=float, default=1e-4, help='learning rate (default: 1e-4)')
parser.add_argument('--cudnn-nondet', action='store_true',
                    help='disable cudnn determinism - might slow down training')
parser.add_argument('--warm-up', type=float, default=0, help='rate of warm up epochs')
parser.add_argument('--no-amp', action='store_true', help='NOT auto mix precision training')#混合精度训练

# network architecture parameters
parser.add_argument('--enc', type=int, nargs='+',
                    help='list of unet encoder filters (default: 16 32 32 32)')#encoder
parser.add_argument('--dec', type=int, nargs='+',
                    help='list of unet decorder filters (default: 32 32 32 32 32 16 16)')#decoder
parser.add_argument('--int-steps', type=int, default=7,
                    help='number of integration steps (default: 7)')#变形场积分步数
parser.add_argument('--int-downsize', type=int, default=2,
                    help='flow downsample factor for integration (default: 2)')#积分下采样因子
parser.add_argument('--bidir', action='store_true', help='enable bidirectional cost function')#双向代价函数，命令行不写为flase，写上为true
parser.add_argument('--model', type=str, default='vm', help='Choose a model to train ')#训练的模型类型


# loss hyperparameters
parser.add_argument('--image-loss', default='ncc',
                    help='image reconstruction loss - can be mse or ncc (default: mse)')#图像损失选择
parser.add_argument('--win', type=int, default=9, help='window size for ncc loss')#NCC损失的窗口大小
parser.add_argument('--lambda', type=float, dest='weight', default=1,
                    help='weight of deformation loss (default: 0.1)')#变形场正则化权重
parser.add_argument('--ignore-label', type=int, nargs='+', default=[0],
                    help='list of ignorable labels')#要忽略的标签列表
parser.add_argument('--cl', type=float, default=0.0, help='whether to use contrastive loss and set its weight')#对比损失权重
# 数据增强参数
parser.add_argument('--use-augmentation', action='store_true', help='enable data augmentation')
parser.add_argument('--aug-rotation', type=float, default=15, help='rotation range for augmentation (degrees)')
parser.add_argument('--aug-scale-min', type=float, default=0.9, help='minimum scale factor')
parser.add_argument('--aug-scale-max', type=float, default=1.1, help='maximum scale factor')
parser.add_argument('--aug-flip-prob', type=float, default=0.5, help='probability of flipping')
parser.add_argument('--aug-spatial-prob', type=float, default=0.8, help='probability of applying spatial augmentation')
parser.add_argument('--aug-intensity', action='store_true', help='enable intensity augmentation')
parser.add_argument('--aug-intensity-prob', type=float, default=0.5, help='probability of applying intensity augmentation')

# 局部刚性配准参数
parser.add_argument('--use-local-rigid', action='store_true', help='enable local rigid registration for each vertebra')
parser.add_argument('--num-vertebrae', type=int, default=4, help='number of vertebrae for local rigid registration')


args = parser.parse_args()
bidir = args.bidir

@contextmanager
def conditional_autocast(enabled: bool):
    if enabled:
        with autocast():
            yield
    else:
        yield

# load and prepare data
with open(args.subj_train, 'rb') as file:
    train_subject = pickle.load(file)
with open(args.subj_val, 'rb') as file:
    val_subject = pickle.load(file)
with open(args.subj_test, 'rb') as file:
    test_subject = pickle.load(file)

data_example = vxm.py.utils.load_volfile(
    args.seg_path + train_subject[0] + '.nii.gz')
inshape = tuple([int(old_size * args.scale) for old_size in data_example.shape])#缩放后的形状
labels_in = np.unique(data_example)#获取分割标签中的标签值[0,1,2,3,4] 0是背景，1-4分别是C1-C4椎骨的标签

if args.chunk:
    assert args.scale == 0.5, "If using chunking opearation, the scale must be 0.5!"
    args.scale *= 2

generator = src_generators.volgen_crossmodality(
    subjects=train_subject,
    vol_path=args.vol_path,
    seg_path=args.seg_path,
    labels=labels_in,
    mode=args.mode,
    same_subject=args.same_subject,
    batch_size=args.batch_size,
    resize_factor=args.scale,
    chunk=args.chunk,#one-hot
)

generator_val = src_generators.volgen_crossmodality(
    subjects=val_subject,
    vol_path=args.vol_path,
    seg_path=args.seg_path,
    labels=labels_in,
    mode=args.mode,
    same_subject=args.same_subject,
    batch_size=args.batch_size,
    resize_factor=args.scale,
    chunk=args.chunk,
)

# no need to append an extra feature axis if data is multichannel
add_feat_axis = not args.multichannel#单通道数据为True，添加特征维度（batch，176，208，192，1）

# prepare model folder
model_dir = args.model_dir
if os.path.exists(model_dir):
    warnings.warn("Ensure that you don't overwrite the former folder!")
# assert not os.path.exists(model_dir), "Ensure that you don't overwrite the former folder!"
os.makedirs(model_dir, exist_ok=True)

# device handling
if args.gpu:
    gpus = args.gpu.split(',')
    nb_gpus = len(gpus)
    device = 'cuda'
    # Note: CUDA_VISIBLE_DEVICES already set before importing torch
    assert np.mod(args.batch_size, nb_gpus) == 0, \
        'Batch size (%d) should be a multiple of the nr of gpus (%d)' % (args.batch_size, nb_gpus)
    # enabling cudnn determinism appears to speed up training by a lot
    torch.backends.cudnn.deterministic = not args.cudnn_nondet
else:
    nb_gpus = 0
    device = 'cpu'

# unet architecture
enc_nf = args.enc if args.enc else [16] * 4
dec_nf = args.dec if args.dec else [16] * 6

# Define a model
if args.model == 'vm':
    model = networks.VxmDense.load(args.load_model, device) \
        if args.load_model else \
        networks.VxmDense(
            inshape=inshape,
            nb_unet_features=[enc_nf, dec_nf],
            bidir=bidir,
            int_steps=args.int_steps,
            int_downsize=args.int_downsize,
        )
elif args.model == 'mm-feat':
    config = CONFIGS_TM['MambaMorph']
    # 更新配置中的积分步数和下采样因子
    config.int_steps = args.int_steps
    config.int_downsize = args.int_downsize
    # 更新局部刚性配准配置
    config.use_local_rigid = args.use_local_rigid
    config.num_vertebrae = args.num_vertebrae
    config.rigid_flow_weight = args.rigid_flow_weight
    config.deform_flow_weight = args.deform_flow_weight
    model = TransMorph.MambaMorphFeat(config)
    if args.load_model:
        model.load_state_dict(torch.load(args.load_model))
    #解冻所有组
    for param in model.parameters():
        param.requires_grad = True
elif args.model == 'mm-swin-feat':
    # MambaMorphSwinFeat: 带特征提取器的Mamba-Swin融合模型
    config = CONFIGS_TM['MambaMorphSwin']
    config.int_steps = args.int_steps
    config.int_downsize = args.int_downsize
    config.img_size = inshape
    # 局部刚性配准配置
    config.use_local_rigid = args.use_local_rigid
    config.num_vertebrae = args.num_vertebrae
    config.rigid_flow_weight = args.rigid_flow_weight
    config.deform_flow_weight = args.deform_flow_weight
    model = TransMorph.MambaMorphSwinFeat(config)
    if args.load_model:
        model.load_state_dict(torch.load(args.load_model))
    # 解冻所有参数
    for param in model.parameters():
        param.requires_grad = True
    print(f"MambaMorphSwinFeat: Fusion of Mamba and SwinTransformer with feature extractor")

if nb_gpus > 1:
    # use multiple GPUs via DataParallel
    model = torch.nn.DataParallel(model)
    model.save = model.module.save

# prepare the model for training and send to device
model.to(device)
transform_model = SpatialTransformer(inshape, mode='bilinear')  # STN
transform_model.to(device)

# prepare image loss
if args.image_loss == 'ncc':
    win = args.win
    image_loss_func = src_loss.NCC(win).loss
elif args.image_loss == 'mse':
    image_loss_func = src_loss.MSE().loss
elif args.image_loss == 'dice':
    image_loss_func = src_loss.Dice().loss
elif args.image_loss == 'mindLoss':
    mind_loss_module = src_loss.MIND_loss()
    mind_loss_module.to(device)
    image_loss_func = mind_loss_module.loss
elif args.image_loss == 'mi':
    image_loss_func = src_loss.MutualInformation().loss
else:
    raise ValueError('Image loss should be "mse" or "ncc", but found "%s"' % args.image_loss)

# need two image loss functions if bidirectional
if bidir:
    losses = [image_loss_func, image_loss_func]
    weights = [0.5, 0.5]
else:
    losses = [image_loss_func]
    weights = [1]

# prepare deformation loss (regularization loss)
losses += [src_loss.Grad('l2', loss_mult=args.int_downsize).loss]
weights += [args.weight]  # Regularization loss

if args.cl > 0.:
    cl_loss_fn = src_loss.ContrastiveSem(scale=0.5)
    cl_weight = args.cl

min_loss = 1e3
train_rec_path = os.path.join(model_dir, 'train_record.txt')
val_rec_path = os.path.join(model_dir, 'val_record.txt')
if os.path.exists(train_rec_path):
    os.remove(train_rec_path)
if os.path.exists(val_rec_path):
    os.remove(val_rec_path)
# TODO: Set hyperparameter
# lr, eps = args.lr, 1e-3
lr, eps = args.lr, 1e-3
r_scale = 5.
reg_weight = 0.001
lowest_loss = 1e3

# TODO: set optimizer
optimizer = torch.optim.Adam(model.parameters(), lr=lr)
if not args.no_amp:
    scaler = torch.cuda.amp.GradScaler()

# set the ignorable labels
if len(args.ignore_label) > 0:
    label_ignore = ~np.isin(labels_in, args.ignore_label)
else:
    label_ignore = np.ones(len(labels_in)).astype(bool)
label_ignore = torch.from_numpy(label_ignore).to(device)

total_step = math.ceil(len(train_subject) / args.batch_size)
if args.chunk:
    args.batch_size *= 8


train_loss_rec = []
val_loss_rec = []
cl_loss_rec = []
image_loss_rec = []  # 记录图像损失（根据选择的loss类型）
grad_loss_rec = []  # 记录梯度正则化损失
val_image_loss_rec = []  # 验证集图像损失
val_grad_loss_rec = []  # 验证集梯度正则化损失

# 获取损失函数的显示名称
loss_display_name = args.image_loss.upper()  # NCC, MSE, DICE, MINDLOSS, MI

# training loops
for epoch in range(args.initial_epoch, args.epochs):
    print(f"Epoch {epoch + 1} / {args.epochs}")
    epoch_loss = []
    epoch_total_loss = []
    step_start_time = time.time()
    train_rec = open(train_rec_path, 'a')
    val_rec = open(os.path.join(model_dir, 'val_record.txt'), 'a')

    model.train()
    if args.cl > 0:
        cl_epoch_rec = []

    for step in range(total_step):
        vols, segs = next(generator)#接受到的是vols归一化后的[src_vols, tgt_vols],segs[src_segs, tgt_segs]独热编码
        

        
        zero_flow = np.zeros(
            (args.batch_size, *tuple([int(tmp / args.int_downsize) for tmp in inshape]), len(inshape)))
        #创建一个全零的形变场，用于计算正则化损失，也就是平滑损失
        inputs = [vols[0], vols[1]]  # src_img, tgt_img
        # y_true = [segs[1], zero_flow]  # tgt_label, 0
        if bidir:
            y_true = [vols[1], vols[0], zero_flow]  # tgt_img, src_img, 0
        else:
            y_true = [vols[1], zero_flow]
        src_label = segs[0]
        src_label = torch.from_numpy(src_label).to(device).float().permute(0, 4, 1, 2, 3)
        inputs = [torch.from_numpy(d).to(device).float().permute(0, 4, 1, 2, 3) for d in inputs]  # volume pairs
        y_true = [torch.from_numpy(d).to(device).float().permute(0, 4, 1, 2, 3) for d in y_true]
        
        # 如果使用局部刚性配准，准备椎骨掩码（将one-hot编码转为label ID）
        if args.use_local_rigid:
            src_mask = torch.argmax(src_label, dim=1, keepdim=False)
        else:
            src_mask = None

        with conditional_autocast(not args.no_amp):
            if args.use_local_rigid:
                ret_dict = model(*inputs, source_mask=src_mask, return_pos_flow=True, return_feature=True)
                warped_vol = ret_dict['moved_vol']
                preint_flow = ret_dict['preint_flow']
                pos_flow = ret_dict['pos_flow']
            else:
                ret_dict = model(*inputs, return_pos_flow=True, return_feature=True)
                warped_vol = ret_dict['moved_vol']
                preint_flow = ret_dict['preint_flow']
                pos_flow = ret_dict['pos_flow']
            # warped_vol, preint_flow, pos_flow = model(*inputs, return_both_flow=True)
            warped_label = transform_model(src_label, pos_flow)#变形后的标签
            # y_pred = (warped_label, preint_flow)#与y_true对应，tgt_label和zero_flow
            if bidir:
                warped_target = ret_dict['moved_target']  # 双向模式下的反向配准图像
                y_pred = (warped_vol, warped_target, preint_flow)
            else:
                y_pred = (warped_vol, preint_flow)
            loss = 0
            loss_list = []
            per_loss = torch.zeros([args.batch_size], device=device)
            for n, loss_function in enumerate(losses):
                # curr_loss = loss_function(y_true[n], y_pred[n], ignore_label=label_ignore)
                curr_loss = loss_function(y_true[n], y_pred[n])
                curr_loss *= weights[n]
                loss_list.append(curr_loss.item())
                loss += curr_loss

            epoch_loss.append(loss_list)
            epoch_total_loss.append(loss.item())
            # print(f"Step: {step}, Loss: {loss}")
            if args.cl > 0:
                feat = ret_dict['feature']
                cl_loss = cl_loss_fn.loss(feat, src_label, y_true[0], ignore_label=label_ignore)
                # print(f"Step: {step}, Loss: {loss}, CL_loss: {cl_loss}")
                loss += cl_loss * cl_weight
                cl_epoch_rec.append(cl_loss.detach().cpu().item())

        optimizer.zero_grad()
        if not args.no_amp:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()

    # print epoch info
    epoch_info = 'Epoch %d/%d' % (epoch + 1, args.epochs)
    epoch_loss_mean = np.mean(epoch_loss, axis=0)  # 计算每个损失分量的平均值
    losses_info = ', '.join(['%.4e' % f for f in epoch_loss_mean])
    loss_info = 'loss: %.4e  (%s)' % (np.mean(epoch_total_loss), losses_info)  # total_loss, sim loss, reg loss
    print(' - '.join((epoch_info, loss_info)), flush=True)
    train_rec.write(f"Epoch {epoch + 1} {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    if args.cl > 0:
        cl_loss_rec.append(np.mean(cl_epoch_rec))
        train_rec.write(f"{round(np.mean(epoch_total_loss), 6)}, {losses_info}, {round(np.mean(cl_epoch_rec), 6)}\n")
    else:
        train_rec.write(f"{round(np.mean(epoch_total_loss), 6)}, {losses_info}\n")
    train_loss_rec.append(np.mean(epoch_total_loss))
    
    # 记录图像损失和grad损失 (如果是双向的话，第0和第1个是图像损失，最后一个是grad；否则第0个是图像损失，第1个是grad)
    if bidir:
        image_loss_rec.append((epoch_loss_mean[0] + epoch_loss_mean[1]) / 2)  # 双向图像损失平均
        grad_loss_rec.append(epoch_loss_mean[2])  # grad损失
    else:
        image_loss_rec.append(epoch_loss_mean[0])  # 图像损失
        grad_loss_rec.append(epoch_loss_mean[1])  # grad损失

    # save model checkpoint
    if epoch % args.steps_per_epoch == 0 and epoch > 0:
        if args.model.startswith('vm'):
            model.save(os.path.join(model_dir, '%04d.pt' % epoch))
        else:
            torch.save(model.state_dict(), os.path.join(model_dir, '%04d.pt' % epoch))

    if np.mean(epoch_total_loss) < lowest_loss:
        if args.model.startswith('vm'):
            model.save(os.path.join(model_dir, 'min_train.pt'))
        else:
            torch.save(model.state_dict(), os.path.join(model_dir, 'min_train.pt'))
        lowest_loss = np.mean(epoch_total_loss)

    # validating loops
    with torch.no_grad():
        model.eval()
        val_total_loss = []
        val_loss = []

        for val_step in range(math.ceil(len(val_subject) / args.batch_size)):
            # generate inputs (and true outputs) and convert them to tensors
            vols, segs = next(generator_val)
            inputs = [vols[0], vols[1]]  # src_img, tgt_img
            # y_true = [segs[1], zero_flow]  # tgt_label, 0
            if bidir:
                y_true = [vols[1], vols[0], zero_flow]  # tgt_img, src_img, 0
            else:
                y_true = [vols[1], zero_flow]
            src_label = segs[0]
            src_label = torch.from_numpy(src_label).to(device).float().permute(0, 4, 1, 2, 3)
            inputs = [torch.from_numpy(d).to(device).float().permute(0, 4, 1, 2, 3) for d in inputs]  # volume pairs
            y_true = [torch.from_numpy(d).to(device).float().permute(0, 4, 1, 2, 3) for d in y_true]
            
            # 如果使用局部刚性配准，准备椎骨掩码
            if args.use_local_rigid:
                src_mask = torch.argmax(src_label, dim=1, keepdim=False)
            else:
                src_mask = None

            # run inputs through the model to produce a warped image and flow field
            with conditional_autocast(not args.no_amp):
                
                if args.use_local_rigid:
                    ret_dict = model(*inputs, source_mask=src_mask, return_pos_flow=True)
                    warped_vol = ret_dict['moved_vol']
                    preint_flow = ret_dict['preint_flow']
                    pos_flow = ret_dict['pos_flow']
                else:
                    ret_dict = model(*inputs, return_pos_flow=True)
                    warped_vol = ret_dict['moved_vol']
                    preint_flow = ret_dict['preint_flow']
                    pos_flow = ret_dict['pos_flow']
                warped_label = transform_model(src_label, pos_flow)
                # y_pred = (warped_label, preint_flow)
                if bidir:
                    warped_target = ret_dict['moved_target']  # 双向模式下的反向配准图像
                    y_pred = (warped_vol, warped_target, preint_flow)
                else:
                    y_pred = (warped_vol, preint_flow)
                # calculate total loss
                loss_val = 0
                loss_list_val = []
                for n, loss_function in enumerate(losses):
                    # curr_loss = loss_function(y_true[n], y_pred[n], ignore_label=label_ignore) * weights[n]
                    curr_loss = loss_function(y_true[n], y_pred[n]) * weights[n]
                    loss_list_val.append(curr_loss.item())
                    loss_val += curr_loss
                val_loss.append(loss_list_val)
                val_total_loss.append(loss_val.item())

        val_loss_mean = np.mean(val_loss, axis=0)  # 计算验证集每个损失分量的平均值
        val_losses_info = ', '.join(['%.4e' % f for f in val_loss_mean])
        print(f"---Validation Loss: {round(np.mean(val_total_loss), 6)} "
              f"({val_losses_info})")
        mean_loss = round(np.mean(val_total_loss), 6)
        val_rec.write(f"Epoch {epoch + 1} {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        val_rec.write(f"{mean_loss}, {val_losses_info}\n")
        val_loss_rec.append(mean_loss)
        
        # 记录验证集的图像损失和grad损失
        if bidir:
            val_image_loss_rec.append((val_loss_mean[0] + val_loss_mean[1]) / 2)  # 双向图像损失平均
            val_grad_loss_rec.append(val_loss_mean[2])  # grad损失
        else:
            val_image_loss_rec.append(val_loss_mean[0])  # 图像损失
            val_grad_loss_rec.append(val_loss_mean[1])  # grad损失

        if np.mean(val_total_loss) < min_loss:
            if args.model.startswith('vm'):
                model.save(os.path.join(model_dir, 'min_val.pt'))
            else:
                torch.save(model.state_dict(), os.path.join(model_dir, 'min_val.pt'))
            min_loss = np.mean(val_total_loss)

    time_consumption = round((time.time() - step_start_time) / 60, 3)
    print('-' * 5 + f'This epoch takes {time_consumption} min.\n')
    train_rec.close()
    val_rec.close()

print(f"Minimum training loss is: {lowest_loss}.")

# plot the loss
plt.plot(np.array(train_loss_rec))
plt.title('Train loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.savefig(os.path.join(model_dir, 'train_loss.png'))
plt.close()

plt.plot(np.array(val_loss_rec))
plt.title('Validation loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.savefig(os.path.join(model_dir, 'val_loss.png'))
plt.close()

if args.cl > 0:
    plt.plot(np.array(cl_loss_rec))
    plt.title('Contrastive loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.savefig(os.path.join(model_dir, 'cl_loss.png'))
    plt.close()

# 绘制图像损失曲线（训练集和验证集）
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
ax1.plot(np.array(image_loss_rec), label='Train')
ax1.plot(np.array(val_image_loss_rec), label='Validation')
ax1.set_title(f'{loss_display_name} Loss')
ax1.set_xlabel('Epoch')
ax1.set_ylabel('Loss')
ax1.legend()
ax1.grid(True, alpha=0.3)

# 绘制grad损失曲线（训练集和验证集）
ax2.plot(np.array(grad_loss_rec), label='Train')
ax2.plot(np.array(val_grad_loss_rec), label='Validation')
ax2.set_title('Gradient Regularization Loss')
ax2.set_xlabel('Epoch')
ax2.set_ylabel('Loss')
ax2.legend()
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(os.path.join(model_dir, f'{args.image_loss}_grad_loss.png'), dpi=150)
plt.close()

# 单独绘制图像损失
plt.figure(figsize=(8, 5))
plt.plot(np.array(image_loss_rec), label=f'Train {loss_display_name}', linewidth=2)
plt.plot(np.array(val_image_loss_rec), label=f'Validation {loss_display_name}', linewidth=2)
plt.title(f'{loss_display_name} Loss', fontsize=14)
plt.xlabel('Epoch', fontsize=12)
plt.ylabel('Loss', fontsize=12)
plt.legend(fontsize=10)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(model_dir, f'{args.image_loss}_loss.png'), dpi=150)
plt.close()

# 单独绘制grad损失
plt.figure(figsize=(8, 5))
plt.plot(np.array(grad_loss_rec), label='Train Grad', linewidth=2)
plt.plot(np.array(val_grad_loss_rec), label='Validation Grad', linewidth=2)
plt.title('Gradient Regularization Loss', fontsize=14)
plt.xlabel('Epoch', fontsize=12)
plt.ylabel('Loss', fontsize=12)
plt.legend(fontsize=10)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(model_dir, 'grad_loss.png'), dpi=150)
plt.close()

# final model save
if args.model.startswith('vm'):
    model.save(os.path.join(model_dir, '%04d.pt' % args.epochs))
else:
    torch.save(model.state_dict(), os.path.join(model_dir, '%04d.pt' % epoch))

print('-' * 75)
print('-' * 30 + "Finish training" + '-' * 30)
print('-' * 75)
""""""
# test loops
if test_subject is not None:
    np.random.seed(0)
    first_mod = args.mode.split('>')[0]
    second_mod = args.mode.split('>')[1]

    test_vol_moving = [os.path.join(args.vol_path, item + f"_{first_mod}.nii.gz") for item in test_subject]
    test_vol_fixed = [os.path.join(args.vol_path, item + f"_{second_mod}.nii.gz") for item in test_subject]
    test_seg = [os.path.join(args.seg_path, item + f".nii.gz") for item in test_subject]
    # Use same subject pairing (mr->ct for each subject)
    print("第一个模态名字", test_vol_moving)
    print("第二个模态名字", test_vol_fixed)
    print("相同受试者分割标签名字", test_seg * 2)
    # Prepare the model
    if args.model == 'vm':
        model = networks.VxmDense.load(os.path.join(model_dir, 'min_train.pt'), device)
    elif args.model == 'vm-feat':
        model = networks.VxmFeat.load(os.path.join(model_dir, 'min_train.pt'), device)
    elif args.model.startswith('mm'):
        model.load_state_dict(torch.load(os.path.join(model_dir, 'min_train.pt')))
    model.to(device)
    model.eval()
    transform_model = SpatialTransformer(inshape, mode='nearest')  # STN
    transform_model.to(device)
    dice_means = []
    with torch.no_grad():
        for idx in range(len(test_subject)):
            # read the images and labels from files (same subject)
            moving = vxm.py.utils.load_volfile(test_vol_moving[idx], add_batch_axis=True, np_var='vol',
                                               add_feat_axis=add_feat_axis, resize_factor=args.scale)
            moving = utils.minmax_norm(moving)
            moving_seg = vxm.py.utils.load_volfile(test_seg[idx], add_batch_axis=True,
                                                   np_var='seg', add_feat_axis=add_feat_axis, resize_factor=args.scale)
            fixed = vxm.py.utils.load_volfile(test_vol_fixed[idx], add_batch_axis=True, np_var='vol',
                                              add_feat_axis=add_feat_axis, resize_factor=args.scale)
            fixed = utils.minmax_norm(fixed)
            fixed_seg = vxm.py.utils.load_volfile(test_seg[idx], add_batch_axis=True,
                                                  np_var='seg', add_feat_axis=add_feat_axis, resize_factor=args.scale)

            moving_seg = src_generators.split_seg_global(moving_seg, labels_in)
            fixed_seg = src_generators.split_seg_global(fixed_seg, labels_in)

            # Infer
            input_moving = torch.from_numpy(moving).to(device).float().permute(0, 4, 1, 2, 3)
            input_fixed = torch.from_numpy(fixed).to(device).float().permute(0, 4, 1, 2, 3)
            # predict
            # moved, warp = model(input_moving, input_fixed, registration=True)
            if args.use_local_rigid:
                input_seg = torch.from_numpy(moving_seg).to(device).float().permute(0, 4, 1, 2, 3)
                src_mask = torch.argmax(input_seg, dim=1, keepdim=False)
                ret_dict = model(input_moving, input_fixed, source_mask=src_mask, return_pos_flow=True)
                moved = ret_dict['moved_vol']
                warp = ret_dict['pos_flow']
            else:
                ret_dict = model(input_moving, input_fixed, return_pos_flow=True)
                moved = ret_dict['moved_vol']
                warp = ret_dict['pos_flow']

            # apply transform
            input_seg = torch.from_numpy(moving_seg).to(device).float().permute(0, 4, 1, 2, 3)
            warped_seg = transform_model(input_seg, warp).squeeze()
            # compute volume overlap (dice)
            overlap = vxm.py.utils.dice(np.argmax(warped_seg.cpu().numpy(), axis=0),
                                        np.argmax(fixed_seg, axis=-1).squeeze(), include_zero=True)
            dice_means.append(np.mean(overlap[label_ignore.cpu().numpy()]))

    print('**TEST: Avg Dice %.4f +/- %.4f' % (np.mean(dice_means), np.std(dice_means)))
    with open(os.path.join(model_dir, 'val_record.txt'), 'a') as test_rec:
        test_rec.write(f"TEST   {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        test_rec.write(f"{round(np.mean(dice_means), 6)} +/- {round(np.std(dice_means), 6)}\n")
