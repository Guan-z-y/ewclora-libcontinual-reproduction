# EWC-LoRA 复现实验日志

本目录用于保存 EWC-LoRA 迁移复现实验相关的日志文件。

完整的项目说明、环境配置、运行命令、复现结果对比表、结果差异分析以及迁移过程中遇到的问题与解决方案，请查看仓库根目录的 README.md。

## 日志文件说明

| 文件 | 说明 |
|---|---|
| logs/libcontinual_a100_batch128.log | LibContinual 迁移版在 CIFAR-100 上的主实验日志，A100，batch size 128 |
| logs/official_repo_a100_batch128_reference.log | 原作者官方 repo 在 CIFAR-100 上的参考复现实验日志，A100，batch size 128 |
| logs/libcontinual_batch8_reference.log | LibContinual 迁移版在 batch size 8 设置下的辅助对照实验日志 |

其中，官方 repo 日志仅用于和论文结果、官方实现结果进行对齐分析。
