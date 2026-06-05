# MSR-registration
Spine CT–MRI Rigid–Deformable Hybrid Registration

## Project Structure
```
MSR-registration-main/
├── msr/                            # 核心代码库
│   ├── __init__.py
│   ├── augmentation.py             # 数据增强
│   ├── generators.py               # 数据生成器
│   ├── py/                         # Python 通用工具
│   │   ├── __init__.py
│   │   └── utils.py
│   ├── tf/                         # TensorFlow 实现
│   │   ├── __init__.py
│   │   ├── layers.py
│   │   ├── losses.py
│   │   ├── networks.py
│   │   └── utils/
│   │       ├── __init__.py
│   │       └── utils.py
│   └── torch/                      # PyTorch 实现
│       ├── __init__.py
│       ├── LocalRigidNet.py        # 局部刚性配准网络
│       ├── TransMorph.py           # TransMorph 配准模型
│       ├── XMorpher.py             # XMorpher 配准模型
│       ├── configs_TransMorph.py   # TransMorph 配置
│       ├── layers.py
│       ├── losses.py
│       ├── mamba.py                # Mamba 模块
│       ├── modelio.py              # 模型读写
│       ├── networks.py
│       ├── node.py                 # 节点定义
│       └── utils.py
├── preprocessing/                  # 数据预处理脚本 (A–G)
│   ├── A.py
│   ├── B.py
│   ├── C.py
│   ├── D.py
│   ├── E.py
│   ├── F.py
│   └── G.py
├── torch/                          # 训练与测试入口
│   ├── train_cross.py              # 训练脚本
│   └── test_cross.py               # 测试脚本
├── requirements.txt                # 依赖列表
├── LICENSE
└── README.md
```

# Tutorial
Mamba环境配置可参考教程https://blog.csdn.net/qq_45645368/article/details/141031972

## Acknowledgments
https://github.com/Guo-Stone/MambaMorph
## Paper
https://arxiv.org/abs/2604.27654

## Environment Setup
    # Install dependencies
    plp install -r requirements.txt
## Train
    python train_cross.py --gpu --epochs --batch-size  --model --use-local-rigid --num-vertebrae
## Test
    python test_cross.py --gpu --load-model --model --output-dir --use-local-rigid --num-vertebrae

## Framework
<img width="1865" height="1804" alt="xiu配色+field" src="https://github.com/user-attachments/assets/69ebb608-fd95-4b3c-9d4c-ff80a031ec50" />

## Result
<img width="846" height="457" alt="image" src="https://github.com/user-attachments/assets/4590ad7f-c4db-40ad-90e6-98155fad2eef" />

## Data
<img width="3260" height="1328" alt="数据流程" src="https://github.com/user-attachments/assets/eb424ca5-40f3-4c15-a42c-45d05059ad00" />

The full data processing pipeline and dataset will also be made publicly available.Please click  [Here](https://pan.quark.cn/s/a5ad35418ca9?pwd=YRr4)
to access the  R-D-Reg dataset.

## HN [nnU-Net](https://github.com/mic-dkfz/nnunet) training weights 
We will store the weight in the cloud drive.
## TH [TotalSegmentator](https://github.com/wasserth/totalsegmentator) weights 
We will store the weight in the cloud drive.
## Quark Cloud Drive
    https://pan.quark.cn/s/a5ad35418ca9?pwd=YRr4

