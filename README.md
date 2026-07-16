# EWC-LoRA on LibContinual

本项目将 EWC-LoRA 的核心逻辑重构并接入**LibContinual** 框架，复用了 LibContinual 的配置系统、训练流程、数据流、日志系统和评估接口，复现并迁移实现了论文 **EWC-LoRA / Revisiting Weight Regularization for Low-Rank Continual Learning**。
原论文为 **ICLR 2026 Poster**，研究方向为参数高效持续学习（Parameter-Efficient
Continual Learning, PECL）。

## 1. 方法概述

持续学习中的核心问题是模型在学习新任务时容易遗忘旧任务。EWC 使用 Fisher 信息矩阵估计参数重要性，并对重要参数施加正则约束，从而缓解灾难性遗忘。

EWC-LoRA 将 EWC 的思想引入低秩持续学习场景。方法在 ViT 的 attention 模块中引入 LoRA，并对低秩更新对应的全维权重变化施加 Fisher 加权约束。其核心正则对象不是 LoRA 参数 `A/B` 本身，而是：

```text
delta_W = B @ A
```

在本项目中，LoRA 主要注入 ViT attention 的 K/V 分支，并通过 Fisher 信息矩阵约束新任务带来的低秩权重变化。

## 2. 迁移内容

本项目将原作者代码中的 EWC-LoRA 逻辑迁移到 LibContinual 中，主要涉及文件如下：

| 原作者官方实现 | LibContinual 迁移实现 | 迁移内容 |
|---|---|---|
| methods/ewclora.py | core/model/ewclora.py | 方法类、EWC penalty、Fisher估计、任务状态管理 |
| models/vit_ewclora.py | core/model/backbone/vit_ewclora.py | ViTbackbone、K/VLoRA 注入、LoRA初始化、累积与reset、梯度hook |
| models/net_ewclora.py | core/model/ewclora.py 中的 EWCLoRANet | backbone与多任务分类头池classifier_pool的封装 |
| 官方配置文件 | config/zz_EWCLoRA/*.yaml | CIFAR-100 B10-10-10 实验配置 |
| 官方模型构造逻辑 | core/model/__init__.py、core/model/backbone/__init__.py | 注册新方法和新backbone，接入LibContinual |

迁移后的方法适配了 LibContinual 的标准接口：

| 接口 | 作用 |
|---|---|
| before_task | 更新当前任务编号、类别范围、分类头状态，并设置可训练参数 |
| observe | 完成单个 batch 的训练，计算交叉熵损失和 EWC 正则项 |
| inference | 拼接所有已学习任务的分类头输出，进行 task-agnostic 推理 |
| after_task | 当前任务结束后估计 Fisher，并更新历史重要性矩阵 |
| get_parameters | 将 LoRA 参数和分类头参数划分为不同学习率的参数组 |

核心迁移点包括：

1. 在 ViT attention 的 K/V 分支注入 LoRA，不修改 Q 分支；
2. 区分历史 LoRA 参数和当前任务 LoRA 参数；
3. 当前任务只训练 lora_new_A/B 和当前任务分类头；
4. 任务切换时，将上一任务的 LoRA 更新累积到共享 LoRA；
5. EWC penalty 作用在 delta_W = B @ A 上；
6. 通过 hook 获取 delta_W 的梯度并估计 Fisher；
7. 使用 classifier_pool 维护每个任务的分类头；
8. 通过 LibContinual 的 before_task / observe / inference / after_task 完成训练流程适配。

## 3. 环境配置
推荐使用 Conda 创建独立环境：

    conda create -n ewclora-libcontinual python=3.10 -y
    conda activate ewclora-libcontinual

安装 PyTorch 时，请根据机器的 CUDA 版本选择对应命令。例如 CUDA 11.7环境可使用：

    pip install torch==2.0.1 torchvision==0.15.2 --index-url https://download.pytorch.org/whl/cu117

然后安装其余依赖：

    pip install -r requirements.txt

本项目主实验使用的实际环境如下：

| 项目 | 版本或配置 |
|---|---|
| GPU | NVIDIA A100 |
| CUDA | 11.7, PyTorch CUDA build |
| Python | 3.10 |
| PyTorch | 2.0.1+cu117 |
| torchvision | 0.15.2+cu117 |
| timm | 0.6.7 |
| 操作系统 | Linux x86_64  |

可使用以下命令检查环境：

    python -c "import torch, torchvision, timm; print(torch.__version__, torchvision.__version__,
    timm.__version__)"
    python -c "import torch; print(torch.cuda.is_available())"

## 4. 数据集与实验设置

实验使用 CIFAR-100 的 B10-10-10 类增量设置：

| 项目 | 设置 |
|---|---|
| 数据集 | CIFAR-100 |
| 总类别数 | 100 |
| 初始任务类别数 | 10 |
| 每次增量类别数 | 10 |
| 任务数 | 10 |
| Backbone | ViT-B/16, ImageNet-21K pretrained |
| LoRA rank | 10 |
| lambda | 1e7 |
| gamma | 1.0 |
| epoch | 20 |
| optimizer | Adam |
| batch size | 128 |
| learning rate | 5e-4 |
| classifier learning rate | 5e-3 |

CIFAR-100 数据默认放置在：

    data/cifar100

## 5. 运行命令

进入项目根目录后，运行：

    conda activate ewclora-libcontinual

### 5.1 正式实验

复现 CIFAR-100 B10-10-10 主实验：

    python run_trainer.py --config ewclora-vit-cifar100-b10-10-10-a100 --device 0


### 5.2 快速验证

如果只想快速验证模型注册、配置读取、任务切换和 Fisher 估计流程，可以运行 smoke test：

    python run_trainer.py --config ewclora-vit-cifar100-smoketest --device 0

## 6. 复现结果

### 6.1 主实验结果对比

| 实现 | 数据集 | 设置 | Last / A10 | Avg / Overall Avg |
|---|---|---|---:|---:|
| 论文结果 | CIFAR-100 | 主表结果 | 87.91 | 92.27 |
| 官方 repo 复现 | CIFAR-100 | A100, batch=128, seed=0 | 88.00 | 92.33 |
| LibContinual 迁移版 | CIFAR-100 | A100, batch=128, seed=0 | 84.67 | 91.36 |

LibContinual 迁移版最终日志结果为：

    Last Average Acc = 84.67
    Overall Avg Acc = 91.36
    Overall Frgt = 3.906
    Overall BwT = -2.85
    Time Costs = 8242.19 sec

### 6.2 任务级 accuracy 曲线

官方 repo A100 batch=128：

    [99.50, 96.75, 95.30, 93.58, 92.34, 90.18, 89.57, 89.00, 89.03, 88.00]

LibContinual 迁移版 A100 batch=128：

    [99.60, 96.70, 94.83, 92.50, 91.76, 89.62, 89.03, 87.90, 87.04, 84.67]

LibContinual 迁移版最后一个任务结束后的 per-task accuracy：

    [90.0, 82.5, 85.7, 80.9, 88.1, 78.4, 79.0, 82.2, 92.0, 87.9]

### 6.3 额外 batch=8 对照实验

为了进一步验证迁移代码是否忠实复现官方实现，还在本地 batch=8 设置下比较了官方 repo 与 LibContinual迁移版：

| 实现 | 数据集 | 设置 | Last / A10 | Avg / Overall Avg |
|---|---|---|---:|---:|
| 官方 repo 复现 | CIFAR-100 | batch=8 | 85.07 | 90.59 |
| LibContinual 迁移版 | CIFAR-100 | batch=8 | 85.12 | 90.64 |

batch=8 下两者结果几乎完全一致，说明迁移版的核心算法逻辑与官方实现基本对齐。

## 7. 结果差异分析

综合来看，LibContinual 迁移版已经完整跑通 CIFAR-100 主实验，并且结果接近论文和官方 repo。
官方 repo 在 A100 batch=128 设置下得到 88.00 / 92.33，与论文主表 87.91 / 92.27 基本一致；
LibContinual 迁移版在相同 A100 batch=128 设置下得到 84.67 / 91.36。其中 Overall Avg 与论文结果差距约为 0.9 个百分点，但 Last / A10 低约 3.2 个百分点，说明迁移版整体学习过程有效，但最后阶段遗忘更明显。

从 accuracy 曲线看，迁移版前 7 个阶段与官方 repo 接近：

    Task 0: 99.60 vs 99.50
    Task 1: 96.70 vs 96.75
    Task 2: 94.83 vs 95.30
    Task 3: 92.50 vs 93.58
    Task 4: 91.76 vs 92.34
    Task 5: 89.62 vs 90.18
    Task 6: 89.03 vs 89.57

但后 3 个阶段差距扩大：

    Task 7: 87.90 vs 89.00
    Task 8: 87.04 vs 89.03
    Task 9: 84.67 vs 88.00

因此，A10 差距主要来自后期任务中的遗忘累积。可能原因分析如下：

1. 框架训练流程差异
   官方 repo 使用自定义训练流程，而 LibContinual 使用统一的 before_task / observe / inference / after_task 生命周期。EWC-LoRA 对任务结束后的 Fisher 估计、LoRA 累积和 reset 时机比较敏感，框架生命周期差异可能影响后期任务表现。

2. LoRA merge/reset 时机差异
   为适配 LibContinual，迁移版将 LoRA 的累积与 reset 延迟到下一个 before_task 中执行，以避免当前任务测试前改变模型状态。该设计保证了 LibContinual 测试流程的正确性，但与官方 repo 内部状态更新顺序仍可能存在细微差异。

3. 随机性控制差异
   CIFAR-100 持续学习实验对随机种子、DataLoader shuffle 和数据增强较敏感。官方 repo 与 LibContinual 的随机种子设置和 DataLoader 构造方式并不完全相同，早期较小差异可能在后续任务中逐步累积。

4. 数据加载和评估路径差异
   两个实现都使用 CIFAR-100 B10-10-10 设置，但官方 repo 和 LibContinual 对测试集组织、任务评估和日志统计的实现路径不同，可能造成 task-wise accuracy 和最终平均值的差异。

5. 预训练权重加载路径差异
   官方 repo 和迁移版都使用 ViT 预训练权重，但模型构造和 state_dict 加载路径不完全相同，可能带来轻微初始化差异。

## 8. 迁移过程中遇到的问题与解决方案

### 8.1 Fisher hook 挂错对象
一开始我没有读懂论文做法，按照常规 EWC 做法，直接将 Fisher hook 挂在了 lora_A.weight 和 lora_B.weight 上，导致结果很不理想。因为 EWC-LoRA 的核心不是直接约束 LoRA 参数本身，而是约束低秩更新对应的全维权重变化：

    delta_W = B @ A

解决方法是在 forward 中显式计算 delta_W，并对 delta_w_k_new 和 delta_w_v_new 注册 hook，使用 delta_W 的梯度平方均值估计 Fisher。

### 8.2 忘记 label offset

在 class-incremental 设置中，第 t 个任务的标签仍然是 CIFAR-100 的全局类别编号，但当前任务分类头只输出当前任务的局部类别。一开始我直接使用了全局标签计算交叉熵，出现了类别索引错误。
解决方法是在 observe 和 Fisher 估计中统一使用局部标签：

    local_target = target - known_classes

推理时再将所有已学习任务的分类头输出拼接为全局类别空间。

### 8.3 LoRA merge/reset 时机错误

一开始我在当前任务测试前就执行了 LoRA merge/reset，导致当前任务评估时模型状态被提前改变，与官方 repo 的评估流程不一致。
解决方法是：

1. 当前任务训练结束后先进行测试；
2. 测试后计算 Fisher；
3. 将 LoRA merge/reset 延迟到下一个 before_task 中执行。

这样更符合 LibContinual 的 task 生命周期。

### 8.4 训练参数冻结范围错误

一开始我错误地将 backbone 主干和历史 LoRA 参数设为了可训练，破坏了参数高效持续学习设置，并影响了遗忘行为。
解决方法是在 freeze_network 中只开放以下参数：

    classifier_pool.current_task
    lora_new_A_k
    lora_new_B_k
    lora_new_A_v
    lora_new_B_v

其余 backbone 参数、历史 LoRA 参数和其他任务分类头均保持 frozen。

## 9. 项目文件结构

本项目主要文件结构如下：

    .
    ├── core/
    │   └── model/
    │       ├── ewclora.py
    │       ├── __init__.py
    │       └── backbone/
    │           ├── vit_ewclora.py
    │           └── __init__.py
    ├── config/
    │   └── zz_EWCLoRA/
    │       ├── ewclora-vit-cifar100-b10-10-10.yaml
    │       └── ewclora-vit-cifar100-b10-10-10-a100.yaml
    ├── reproduce/
    │   └── ewclora/
    │       ├── README.md
    │       └── logs/
    │           ├── libcontinual_a100_batch128.log
    │           ├── official_repo_a100_batch128_reference.log
    │           └── libcontinual_batch8_reference.log
    ├── run_trainer.py
    ├── requirements.txt
    └── README.md

其中 reproduce/ewclora/logs/ 保存本项目复现实验相关日志。官方 repo 日志仅作为论文结果对齐参考，不包含原作者完整代码仓库。

## 10. 总结

本项目完成了 EWC-LoRA 从官方实现到 LibContinual 框架的迁移复现，并在 CIFAR-100 B10-10-10 主实验上完成验证。
结果表明，官方 repo 可以在当前环境下复现论文主表结果；LibContinual 迁移版的 Overall Avg 与论文和官方 repo 基本接近，但 Last / A10 存在一定差距，主要来自后期任务遗忘累积。额外 batch=8 对照实验显示迁移版与官方 repo 几乎完全一致，说明核心算法迁移基本正确。
