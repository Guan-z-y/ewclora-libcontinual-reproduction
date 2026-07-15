# EWC-LoRA on LibContinual

## 1. 项目简介

本项目在 **LibContinual** 框架中复现并迁移实现了论文 **EWC-LoRA: Revisiting Weight Regularization for Low-Rank Continual Learning**。论文的核心思想是：在参数高效持续学习（PECL）场景下，不再为每个任务单独分配一套独立 LoRA 模块，而是对**共享的低秩更新**施加 EWC 风格的权重正则，从而在**保持存储和推理开销不随任务数增长**的同时，更好地平衡稳定性与可塑性。

本次复现采取的是**最小必要迁移**策略：不直接提交或照搬官方 repo 的训练外壳，而是仅将方法独有的核心部分迁入 LibContinual，包括：

- `core/model/backbone/vit_ewclora.py`：EWC-LoRA 所需的 ViT backbone 与 LoRA attention 结构；
- `core/model/ewclora.py`：持续学习方法逻辑、任务状态管理、EWC 正则项、Fisher 信息矩阵估计；
- `config/.../ewclora-vit-cifar100-b10-10-10.yaml`：实验配置文件。

本实现已经在 **CIFAR-100 B10-10-10** 设置下完成了：

1. 迁移版 smoke test 跑通；
2. LibContinual 正式实验跑通；
3. 结果与官方 repo 在本机上的正式结果高度对齐。

---

## 2. 论文与任务背景

论文标题：**Revisiting Weight Regularization for Low-Rank Continual Learning**  
会议：**ICLR 2026**

在低秩持续学习中，已有方法常常采用“每个任务一套独立 LoRA 模块”的做法，以减少任务干扰；但这种做法会导致存储开销随任务数线性增长。EWC-LoRA 则尝试用 **EWC 风格的重要性加权正则**来约束共享低秩更新，从而在保持常数级存储与推理开销的前提下缓解遗忘。论文主结果显示，在 CIFAR-100 上，EWC-LoRA 的性能达到 **A10 = 87.91，Avg = 92.27**，并且在稳定性-可塑性权衡上优于多种 LoRA 类方法。

---

## 3. 迁移思路

### 3.1 为什么不需要迁移整个官方 repo

LibContinual 已经提供了通用训练器、配置系统、数据流、日志系统和验证流程，因此迁移时不需要把官方 repo 的 `main.py`、`trainer.py`、`dataloader`、`utils` 整体搬进来；否则就失去了“迁移到框架”的意义。LibContinual 的 `run_trainer.py` 已经负责读取配置并启动训练，而 `Trainer` 已经负责 task 循环、调用 `before_task / observe / inference / after_task`、以及统计最终指标。

因此，本项目采用的是：

- **框架层复用**：训练外壳、日志、优化器初始化、验证与总体统计全部交给 LibContinual；
- **方法层迁移**：仅迁移 EWC-LoRA 独有的 backbone、EWC 正则、Fisher 估计、任务切换逻辑。

### 3.2 迁移了哪些代码板块

#### （1）Backbone：`vit_ewclora.py`

该文件实现了：

- `AttentionEWCLoRA`：在 ViT attention 的 K/V 投影中注入 LoRA；
- 旧 LoRA 与新任务 LoRA 两套参数；
- `delta_w_k_new_grad / delta_w_v_new_grad` 梯度 hook，用于后续 Fisher 估计；
- `accumulate_and_reset_lora()`：任务结束后将新 LoRA 累加到共享 LoRA，并重置新 LoRA；
- `ViTEWCLoRA`：支持加载 `vit_base_patch16_224_in21k` 预训练模型。

#### （2）方法类：`ewclora.py`

该文件实现了：

- `EWCLoRANet`：将 backbone 与多任务分类头池 `classifier_pool` 组合起来；
- `EWCLoRA`：实现 `before_task / observe / inference / after_task / get_parameters` 等 LibContinual 所需接口；
- `FisherComputer`：在低秩更新对应的全维权重空间上估计 Fisher 信息矩阵。

#### （3）配置：`ewclora-vit-cifar100-b10-10-10.yaml`

配置文件将实验设定接入 LibContinual，包括：

- `dataset: binary_cifar100`
- `backbone.name: vit_ewclora`
- `classifier.name: EWCLoRA`
- `epoch: 20`
- `batch_size: 8`
- `lr: 5e-4`
- `fc_lrate: 5e-3`
- `gamma: 1.0`
- `lambda: 1e7`
- `task_num: 10`

### 3.3 为什么使用 `binary_cifar100`

为了尽量与官方 CIFAR-100 设置保持一致，并减少自定义图片文件夹格式带来的偏差，本实现使用 `binary_cifar100` 作为 LibContinual 侧的数据集配置。

### 3.4 一个关键的时序适配：LoRA 的延迟累积

在官方 repo 中，`after_task()` 的执行时机与 LibContinual 略有不同。为了尽量保持评估顺序与官方实现一致，本项目没有在当前任务测试之前立刻 merge/reset LoRA，而是将 LoRA 的真正累积延迟到下一个 `before_task()` 中执行。这样可以避免由于框架调用顺序差异导致的评估偏移。

---

## 4. 环境配置

### 4.1 推荐环境

建议使用单独的 conda 环境，例如：

```bash
conda create -n libcontinual python=3.10 -y
conda activate libcontinual
```

### 4.2 核心依赖

本项目至少需要以下依赖：

- `torch`
- `torchvision`
- `timm`
- `numpy`
- `pyyaml`
- `tqdm`

本地成功运行时，`timm` 必须安装在**当前激活的 Python 环境**里；如果环境切错，会首先在 `import timm` 时报错。

### 4.3 环境自查命令

在运行前建议先检查当前解释器：

```bash
python -c "import sys; print(sys.executable)"
python -c "import torch, timm; print(torch.__version__, timm.__version__)"
```

如果这两条命令不能正常执行，请先修正环境。

### 4.4 数据准备

CIFAR-100 会在首次运行时自动下载到：

```text
./data/cifar100
```

---

## 5. 目录与关键文件说明

本次迁移相关的关键文件如下：

```text
LibContinual/
├─ core/
│  ├─ model/
│  │  ├─ ewclora.py
│  │  └─ backbone/
│  │     └─ vit_ewclora.py
├─ config/
│  └─ .../
│     ├─ ewclora-vit-cifar100-smoketest.yaml
│     └─ ewclora-vit-cifar100-b10-10-10.yaml
├─ log/
│  └─ EWCLoRA/
│     ├─ binary_cifar100..vit_ewclora--ep1--s0__....log
│     └─ binary_cifar100..vit_ewclora--ep20--s0__....log
└─ run_trainer.py
```

LibContinual 的日志文件会被写到：

```text
<save_path>/log/<classifier_name>/
```

本项目里 `save_path` 为空字符串，因此日志会直接落到项目目录下的 `log/EWCLoRA/`。

---

## 6. 运行命令

### 6.1 迁移版 smoke test

先做一个 3-task、1-epoch 的 smoke test，验证：

- 配置是否能被正确读取；
- backbone / classifier 是否能被正确注册；
- Fisher 流程是否能跑通；
- 训练、验证、最终统计是否完整。

运行命令：

```bash
conda activate libcontinual
cd /d D:\cl_project\LibContinual
python run_trainer.py --config ewclora-vit-cifar100-smoketest
```

该 smoke test 的实际配置为：

- `dataset = binary_cifar100`
- `task_num = 3`
- `epoch = 1`
- `batch_size = 8`
- `fisher_max_batches = 2`
- `lr = 5e-4`
- `fc_lrate = 5e-3`

### 6.2 正式复现实验

正式实验命令如下：

```bash
conda activate libcontinual
cd /d D:\cl_project\LibContinual
python run_trainer.py --config ewclora-vit-cifar100-b10-10-10
```

该配置实际对应：

- `dataset = binary_cifar100`
- `task_num = 10`
- `init_cls_num = 10`
- `inc_cls_num = 10`
- `epoch = 20`
- `batch_size = 8`
- `backbone = vit_ewclora`
- `classifier = EWCLoRA`
- `rank = 10`
- `gamma = 1.0`
- `lambda = 1e7`
- `optimizer = Adam`
- `scheduler = CosineAnnealingLR`

---

## 7. 复现结果对比表

### 7.1 论文结果、官方 repo 本机结果、LibContinual 迁移结果

| 实现 | 数据集 | 设置 | Last / A10 | Avg |
|---|---|---|---:|---:|
| 论文结果 | CIFAR-100 | rank=10 | 87.91 | 92.27 |
| 官方 repo（本机） | CIFAR-100 | `epochs=20, batch_size=8, seed=0` | 85.07 | 90.59 |
| LibContinual 迁移版 | binary_cifar100 | `epochs=20, batch_size=8, seed=0` | 85.12 | 90.64 |

### 7.2 结果解读

最重要的结论不是“和论文绝对数值是否完全一致”，而是：

**LibContinual 迁移版与官方 repo 本机结果几乎完全对齐。**

两者差值只有：

- Last / A10：`85.12 - 85.07 = +0.05`
- Avg：`90.64 - 90.59 = +0.05`

这说明迁移实现本身是正确的，方法逻辑没有在接入 LibContinual 的过程中被破坏。

与论文结果相比，本地结果略低，这是可以解释的：官方论文主表中的结果通常是在作者原始设定、多 seed、以及更强实验条件下得到的，而本地复现为了适配显存与运行环境，采用了更保守的 batch size 和工程配置。论文中 CIFAR-100 的 rank=10 主结果为 87.91 / 92.27，本地两份实现均低约 2 个点左右，但彼此高度一致。

---

## 8. Smoke test 结果

在正式长跑前，本项目先进行了 smoke test：

- `task_num = 3`
- `epoch = 1`
- `fisher_max_batches = 2`

最终结果为：

- `Last Average Acc = 91.40`
- `Overall Avg Acc = 94.32`
- `Forgetting = 3.400`
- `Backward Transfer = -2.27`

这一步的作用不是对齐论文，而是验证迁移主干是否跑通。事实证明，迁移版在 3-task + 1-epoch 下能够顺利完成训练、验证、统计与 Fisher 流程，因此后续正式实验是建立在一个已验证的迁移实现之上的。

---

## 9. 指标说明

LibContinual 的最终日志会输出：

- `[Batch] Last Average Acc`
- `[Batch] Overall Avg Acc`
- `Overall Frgt`
- `Overall BwT`
- `Average Acc Table`

其中可以粗略理解为：

- **Last Average Acc**：学完最后一个任务后的最终平均性能；
- **Overall Avg Acc**：整个任务序列上的平均性能；
- **Overall Frgt**：总体遗忘程度；
- **Overall BwT**：后向迁移程度；
- **Average Acc Table**：任务级性能矩阵。

---

## 10. 遇到的坑及解决方案

### 10.1 `timm` 导入报错

**现象**  
运行 `python run_trainer.py ...` 时，先在 `import timm` 处报错。

**原因**  
当前 shell 绑定到了错误的 Python 解释器，`timm` 没装在当前环境里。

**解决办法**  
切回之前已经跑通 LibContinual 的 conda 环境，再检查：

```bash
python -c "import sys; print(sys.executable)"
python -c "import torch, timm; print(torch.__version__, timm.__version__)"
```

只要这两条能通过，`timm` 本身就不是问题。

---

### 10.2 官方 repo 与 LibContinual 的执行顺序不同

**现象**  
即使方法逻辑迁得差不多，结果仍然可能与官方 repo 有较大偏差。

**原因**  
LibContinual 的 `after_task()` 调用时机与官方 repo 不完全一致；如果在错误时刻直接 merge/reset LoRA，会导致测试使用到的模型状态发生偏移。

**解决办法**  
将 LoRA 的真正累积延迟到下一个 `before_task()` 中执行，以尽量保持评估顺序与官方实现一致。

---

### 10.3 结果和官方 repo 差很多

**优先排查项**

1. 类别顺序是否一致；
2. 数据集是否使用 `binary_cifar100`；
3. `optimizer lr` 与 `classifier lr` 是否对齐；
4. 评估顺序是否与官方逻辑一致。

---

### 10.4 显存与运行时间问题

官方仓库初始实验设置较激进，而本地为了适配单卡显存与 Windows 环境，官方 repo 与 LibContinual 迁移版的正式运行都采用了：

- `epochs = 20`
- `batch_size = 8`
- `num_workers = 0`

这种缩减不会改变方法本身，但会让训练更慢。本地 LibContinual 正式版总耗时约 **52960.62 秒**，即约 **14.7 小时**。

---

## 11. 日志与输出文件

正式实验日志位于：

```text
log/EWCLoRA/binary_cifar100..vit_ewclora--ep20--s0__YYYY-MM-DD_HH-MM.log
```

smoke test 日志位于：

```text
log/EWCLoRA/binary_cifar100..vit_ewclora--ep1--s0__YYYY-MM-DD_HH-MM.log
```

如果还需要画图，可以从日志中提取：

- 每个 task 的 `Last Average Acc`
- 每个 task 内部每个 epoch 的 `Loss`
- 每个 task 内部每个 epoch 的 `Average Acc`

---

## 12. 总结

本项目成功将 **EWC-LoRA** 从官方实现迁移到了 **LibContinual** 框架中，并在 **CIFAR-100 B10-10-10** 设置下完成了从 smoke test 到正式实验的完整验证。迁移版在 LibContinual 中的正式结果为：

- **Last Average Acc = 85.12**
- **Overall Avg Acc = 90.64**

与官方 repo 在本机上的正式结果：

- **85.07 / 90.59**

几乎完全对齐。

这说明本次迁移不仅“能跑”，而且已经在结果上验证了核心逻辑的正确性。对于课程项目 Level-1 而言，这已经属于较高完成度的复现与框架集成。

---

## 13. 深度思考

### 13.1 为什么这次迁移代码量不算大，但依然是完整迁移

这次项目最值得反思的一点是：  
**“迁移到框架”并不等于“把官方 repo 全部复制一遍”。**

真正高质量的框架迁移，应当只迁移**方法独有的最小必要部分**：

- backbone 中真正与方法相关的结构改动；
- 方法类中的状态管理、损失函数、正则项、Fisher 估计；
- 与框架接口兼容的配置。

其余像训练器、日志器、数据流和指标统计，本来就是框架应当复用的部分。  
因此，代码量少并不代表迁移浅；恰恰相反，说明迁移做到了**解耦**和**最小实现**。

### 13.2 为什么最终对齐“官方 repo 本机结果”比直接追论文主表更重要

论文主表中的数值当然重要，但对一次工程复现来说，更关键的是：

1. 官方实现能否在本机跑通；
2. 迁移版能否在同一机器上复现出与官方实现几乎一致的结果。

如果连这一步都做不到，那么即使某次偶然撞到接近论文值，也无法证明迁移是正确的。  
本项目首先拿到了官方 repo 的本机参考结果 85.07 / 90.59，再让 LibContinual 迁移版逼近到 85.12 / 90.64，这种“先本机对齐、再讨论与论文差距”的策略，是更可信、更可解释的复现方式。

### 13.3 对持续学习方法复现的认识

持续学习复现的难点，不仅在于训练是否能进行，更在于：

- task 边界逻辑是否正确；
- old/new class 的标签偏移是否正确；
- 评估顺序是否一致；
- 遗忘与后向迁移指标是否按同一口径统计。

因此，真正决定复现质量的，往往不是“模型 forward 写得像不像”，而是**任务切换、状态更新、评估顺序、数据顺序**这些看似工程细节的部分。  
本项目最深的体会就是：**持续学习方法的复现，本质上既是算法问题，也是系统工程问题。**

---

## 14. 参考文献

```bibtex
@article{zheng2026revisiting,
  title={Revisiting Weight Regularization for Low-Rank Continual Learning},
  author={Zheng, Yaoyue and Zhang, Yin and van de Weijer, Joost and van de Ven, Gido M and Du, Shaoyi and Zhang, Xuetao and Tian, Zhiqiang},
  journal={arXiv preprint arXiv:2602.17559},
  year={2026}
}
```