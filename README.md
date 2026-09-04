# `crossec_forecast`

> **High-Performance Cross-Sectional & Time-Series Financial Forecasting and Model Benchmarking Framework**

`crossec_forecast` 是一个专门针对标准化金融截面时序面板（如 [`cross_section_enrich`](https://github.com/GeekChomolungma/cross_section_enrich) 生成的 `standar_panel.csv`）设计的量化神经网络训练与模型横评对比框架。

---

## 🌟 核心特性

1. **金融面板专用通用 DataLoader**
   - 原生支持 `(timestamp asc, symbol asc)` 双重排序结构。
   - 自适应滑动窗口（默认 $L=6$ 回看步长），自动兼容新老标的上市时间不同（Ragged Start）且无虚假 0 填充。
   - 自动过滤尾部未来未揭露标签（`target` 为 `NaN`），杜绝未来信息泄露。
   - 时序无泄露切分器（`TimeSplitter`），支持在 Train / Val / Test 切分点之间设置 `embargo` 隔离带。

2. **热插拔模型插件体系 (Plugin / Plug-out Registry)**
   - 统一继承 `BaseClassifierModel`，使用 `@register_model("name")` 装饰器注册。
   - 预置模型插件：
     - `MLPClassifier`: 展平聚合超大向量前馈网络。
     - `LSTMClassifier`: 时序循环网络（支持 Last/Mean/Attention Pooling）。
     - `DLinearClassifier`: 经典轻量趋势/季节分解时序模型。
   - v1 只做 from-scratch 训练（随机初始化，仅在自有样本上训练），暂不接入外部预训练权重。
   - 预留 `PretrainedBackboneModel` 接口（`crossec_forecast/models/pretrained.py`）：未来接入 Chronos2 等预训练时序基础模型时，继承它、加载真实权重、用 `@register_model(...)` 注册即可复用同一套训练/评测/sweep 流程，无需改动其他代码。
   - 新增模型只需单独新建一个 `.py` 文件并注册，无需改动训练器与评测逻辑。

3. **量化三阶段严谨训评范式 (Three-Stage Quant Lifecycle)**
   - **Stage 1 (Train)**：连续可微损失（BCE / FocalLoss）与梯度反向传播，平滑塑造高维特征流形。
   - **Stage 2 (Validation)**：剥离 Loss 假象，以截面 **Mean Rank IC 均值**（或 ICIR）作为早停与最佳 Checkpoint 锁定的唯一标尺。
   - **Stage 3 (Test / OOS)**：测试集全套指标评估（Test Rank IC, ICIR, AUC, F1, Top-Bottom Spread），并预留 `BaseBacktestEvaluator` 真实交易摩擦与策略回测接口。

4. **一键多模型横评对比器 (`BenchmarkEngine`)**
   - 在相同数据切分下，批量调度多模型训练与测试。
   - 自动导出 Markdown、CSV 和 JSON 格式的多维性能对比表。

---

## 🚀 快速开始

```python
from crossec_forecast import (
    DataConfig,
    TrainConfig,
    BenchmarkConfig,
    build_dataloaders,
    BenchmarkEngine,
)

# 1. 加载数据并构建 DataLoader (L=6)
data_config = DataConfig(
    target_col="logret1_win",
    fwd_ret_col="fwd_logret_1",
    seq_len=6,
    train_ratio=0.70,
    val_ratio=0.15,
    test_ratio=0.15,
    embargo_steps=1,
    batch_size=64,
)

train_loader, val_loader, test_loader, meta_info = build_dataloaders(
    data="standar_panel.csv",
    config=data_config,
)

# 2. 配置训练器 (基于 Val Rank IC 自动早停)
train_config = TrainConfig(
    epochs=20,
    lr=1e-3,
    early_stopping_patience=5,
)
# loss 由模型自己拥有：BaseClassifierModel.compute_loss 从 model config 读
# loss_type ("bce"|"focal") / focal_gamma / focal_alpha；Trainer 对 loss 无感知。

# 3. 配置待横评的模型清单
benchmark_config = BenchmarkConfig(
    models=[
        {"name": "mlp", "config": {"hidden_dims": [64, 32], "dropout": 0.2}},
        {"name": "lstm", "config": {"hidden_dim": 48, "num_layers": 2}},
        {"name": "dlinear", "config": {"kernel_size": 3}},
        # 未来接入预训练骨干（如 Chronos2）时，在这里加一行 {"name": "<registered_name>", "config": {...}}
    ],
    export_dir="./benchmark_reports",
)

# 4. 一键执行横向横评
engine = BenchmarkEngine(
    train_loader=train_loader,
    val_loader=val_loader,
    test_loader=test_loader,
    meta_info=meta_info,
    train_config=train_config,
    benchmark_config=benchmark_config,
)

summary_df = engine.run()
```

---

## 📚 深入文档

README 只覆盖快速上手，更完整的内容拆分在两份文档：

- **[架构设计 · IMPLEMENTATION_PLAN.md](./IMPLEMENTATION_PLAN.md)** —— 端到端系统架构、三阶段训评范式、数据集规范、库内目录与模块划分。
- **[实验工作流 · EXPERIMENTS.md](./EXPERIMENTS.md)** —— `experiment.yaml` 配置体系、`pipelines/` 与 `scripts/` 用法、单模型训练 / 推理 / 多模型横评 / 集群批量调度、wandb 接入。

---

## 🔌 扩展自定义模型 (Plugin a Custom Model)

扩展一个新模型极为简单：

```python
import torch
import torch.nn as nn
from crossec_forecast.models import BaseClassifierModel, register_model

@register_model("my_custom_model")
class MyCustomModel(BaseClassifierModel):
    def __init__(self, config):
        super().__init__(config)
        # self.seq_len, self.feature_dim, self.cov_dim, self.num_classes 自动注入
        # x 是打包好的 [B, seq_len, feature_dim(+cov_dim)] 单一张量：需要协变量的模型自己切
        # x[..., :self.feature_dim] / x[..., self.feature_dim:self.feature_dim+self.cov_dim]
        self.conv = nn.Conv1d(self.feature_dim, 32, kernel_size=3, padding=1)
        self.fc = nn.Linear(32 * self.seq_len, self.num_classes)

    def forward(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
        # x shape: [B, seq_len, feature_dim]
        x_trans = x.permute(0, 2, 1) # [B, feature_dim, seq_len]
        h = torch.relu(self.conv(x_trans))
        logits = self.fc(h.reshape(x.size(0), -1))
        return logits # [B, 1]
```

定义后即可直接在 `BenchmarkConfig` 中使用 `"my_custom_model"` 参与横评！

