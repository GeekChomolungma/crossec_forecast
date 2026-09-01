# 实验工作流 (Experiment Workflow)

> 面向实盘 / 集群的操作层。库本身（`crossec_forecast/`）提供原语，这一层把它们组织成
> **由单一 YAML 驱动、脚本只有十来行** 的可复现实验流程。

- [1. 设计原则](#1-设计原则)
- [2. 目录与产物](#2-目录与产物)
- [3. 配置体系](#3-配置体系)
  - [3.1 `experiment.yaml` 结构](#31-experimentyaml-结构)
  - [3.2 合并优先级与覆盖方式](#32-合并优先级与覆盖方式)
  - [3.3 片段组合与插值](#33-片段组合与插值)
- [4. 三条主流程](#4-三条主流程)
  - [4.1 单模型训练 / 验证 / 测试](#41-单模型训练--验证--测试)
  - [4.2 推理](#42-推理)
  - [4.3 多模型横评](#43-多模型横评)
- [5. 批量 / 集群调度](#5-批量--集群调度)
- [6. Run 目录布局](#6-run-目录布局)
- [7. wandb 接入](#7-wandb-接入)
- [8. 编程接口](#8-编程接口)
- [9. 注意事项](#9-注意事项)

---

## 1. 设计原则

| 原则 | 落地 |
| --- | --- |
| **单一事实源** | 一次 run 的全部配置集中在一份 `experiment.yaml`，不散落到命令行 |
| **脚本极薄** | `scripts/*.py` 只做 “解析 `-c` + 覆盖串 → 调 pipeline”，无业务逻辑 |
| **逻辑可复用** | 真正的流程在 `crossec_forecast/pipelines/`，可被 import、被测试、被其它服务调用 |
| **强 schema** | `ExperimentConfig` 结构化 schema 提供类型与默认值；YAML 里写错 key 直接报错 |
| **可复现** | 每次 run 把解析后的完整 config 落盘到 `runs/<exp>/<run>/config.yaml`，可原样重投 |
| **库 / 应用分离** | `crossec_forecast/` 是库；`scripts/` + `experiments/` 是使用该库的应用代码 |

---

## 2. 目录与产物

```text
experiments/
├── experiment.yaml           # 规范全量配置（唯一事实源）
├── models/                    # 可选的每模型片段，-c 叠加在 base 之上
│   ├── lstm.yaml
│   ├── mlp.yaml
│   ├── dlinear.yaml
│   └── pretrained.yaml.example   # 模板：未来接入预训练骨干（如 Chronos2）时参考，目前不可直接跑
└── sweep.example.yaml         # 批量调度规格（变体 × 网格）

crossec_forecast/
├── configs/experiment_schema.py   # ExperimentConfig 结构化 schema + 类型默认值
└── pipelines/                     # 可 import 的流程逻辑，无 CLI
    ├── context.py                 # load_config / load_experiment / schema→dataclass 适配器
    ├── train_pipeline.py          # run_train
    ├── infer_pipeline.py          # run_infer
    ├── benchmark_pipeline.py      # run_benchmark
    └── tracking.py                # WandbTracker

scripts/
├── train.py                  # 单模型训练入口
├── infer.py                  # 推理入口
├── benchmark.py              # 多模型横评入口
├── sweep.py                  # 批量 / 集群调度器
├── run_all_models.sh         # 本地顺序跑完全部 baseline
└── slurm/
    └── train.sbatch.tmpl     # 唯一的集群作业定义；sweep.py --launcher slurm 逐 job 渲染

runs/                         # 所有产物（已 gitignore）
├── <experiment.name>/<run.name>/   # 每次训练 / 推理 / 横评的实例与结果
└── _sweeps/<时间戳>_<spec名>/       # 一次 sweep 调度的账本（configs/ + sbatch/）
```

---

## 3. 配置体系

### 3.1 `experiment.yaml` 结构

schema 与默认值定义在 [`crossec_forecast/configs/experiment_schema.py`](./crossec_forecast/configs/experiment_schema.py)。

| 段 | 关键字段 | 说明 |
| --- | --- | --- |
| `experiment` | `name`, `seed` | 逻辑实验名（归组 run 与 wandb run）、全局随机种子 |
| `run` | `name`, `output_root`, `tags` | run 文件夹名（`null` 时自动 `<时间戳>_<模型>_<hash>`）、产物根目录 |
| `data` | `path`, `target_col`, `fwd_ret_col`, `timestamp_col`, `symbol_col`, `feature_pattern`, `feature_cols`, `seq_len`, `split.*`, `batch_size`, `shuffle_train`, `num_workers`, `drop_last` | 数据路径、列名、特征列选取（正则或显式白名单）、回看窗口 `L`、时序切分比例与 embargo、DataLoader 参数 |
| `train` | `epochs`, `lr`, `weight_decay`, `grad_clip_norm`, `early_stopping_patience`, `min_delta`, `device`, `scheduler_*` | 训练与优化超参；早停以 val mean rank IC 为准。**loss 不在这里**——由模型自己拥有（`model.compute_loss`），`loss_type` / `focal_*` 放进 `model.config` |
| `model` | `name`, `config` | **单模型**（`train` / `infer` 用）：注册名 + 该模型特异化参数字典（含 `loss_type` / `focal_*`） |
| `benchmark` | `top_quantile`, `models[]` | **模型清单**（`benchmark` 用）：每项 `{name, config}` |
| `wandb` | `enabled`, `project`, `entity`, `mode`, `group`, `job_type`, `name`, `notes`, `tags`, `log_freq` | 实验追踪 |

`data` / `train` 段与库内既有的 `DataConfig` / `TrainConfig` 字段一一对应，
[`pipelines/context.py`](./crossec_forecast/pipelines/context.py) 里的适配器只是逐字段拷贝。

### 3.2 合并优先级与覆盖方式

```
schema 默认值   <   一个或多个 YAML 文件   <   CLI dot-list 覆盖
```

每个脚本只认两类参数：

| 形式 | 含义 |
| --- | --- |
| `-c / --config PATH` | 实验 YAML；**可重复**，片段从左往右合并 |
| `KEY=VALUE ...`（位置参数） | OmegaConf dot-list 覆盖，如 `train.lr=0.0005 model.name=lstm` |

```bash
python scripts/train.py -c experiments/experiment.yaml \
    model.name=dlinear train.lr=0.0005 data.seq_len=8 wandb.enabled=false
```

写错 key（如 `trian.lr=0.1`）会因 schema struct 模式直接报错，而不是被静默忽略。

### 3.3 片段组合与插值

**片段组合** —— base 配置 + 每模型片段：

```bash
python scripts/train.py -c experiments/experiment.yaml -c experiments/models/lstm.yaml
```

`experiments/models/lstm.yaml` 只覆盖 `model:` 段，其余继承 base。

**插值** —— YAML 内可用 `${...}` 引用其它键，避免重复：

```yaml
wandb:
  group: ${experiment.name}
```

---

## 4. 三条主流程

### 4.1 单模型训练 / 验证 / 测试

```bash
python scripts/train.py -c experiments/experiment.yaml
python scripts/train.py -c experiments/experiment.yaml -c experiments/models/lstm.yaml
python scripts/train.py -c experiments/experiment.yaml model.name=dlinear train.lr=0.0005
```

`run_train` 依次完成：构建 DataLoader → `build_model` → `Trainer.fit`（基于 val rank IC 早停，
每 epoch 通过 `callbacks` 钩子上报 wandb）→ OOS 测试 + `SimpleLongShortBacktester` 回测 →
落盘 `metrics.json` / `history.json` / `test_predictions.csv` / best checkpoint。

### 4.2 推理

用训练好的 checkpoint 对**整块 panel**顺序打分（内部构建单个
`PanelTimeSeriesDataset(is_inference=True)`，不做训练/验证/测试三切分，保留所有已揭露样本）：

```bash
# 复用某次 run 落盘的完整 config
python scripts/infer.py -c runs/baseline_v1/<run>/config.yaml \
    --checkpoint runs/baseline_v1/<run>/checkpoints/lstmclassifier_best.pt

# 对一份全新 panel 打分（模型 / 特征规格来自 config）
python scripts/infer.py -c experiments/experiment.yaml -c experiments/models/lstm.yaml \
    --checkpoint path/to/lstm_best.pt --data ./data/new_panel.csv --out preds_2026Q3.csv
```

输出 `runs/<exp>/<run>/<--out>`，列为 `timestamp, symbol, pred_prob, fwd_logret_1`。

### 4.3 多模型横评

```bash
python scripts/benchmark.py -c experiments/experiment.yaml
python scripts/benchmark.py -c experiments/experiment.yaml benchmark.top_quantile=0.1
```

在相同数据切分下对 `benchmark.models[]` 逐个训练 + 测试，
导出 `runs/<exp>/<run>/reports/benchmark_summary.{csv,md,json}`，按 test rank IC 降序。

---

## 5. 批量 / 集群调度

`scripts/sweep.py` 把一份 sweep 规格展开成 **N 份独立完整的 config**，再按 launcher 派发。

**一个 job = 一个 `variant` × 一个 `grid` 点**，其覆盖串为
`common_overrides + variant.overrides + 该 grid 点`。

`experiments/sweep.example.yaml`：

```yaml
base_config: experiments/experiment.yaml
entry: train                     # train | benchmark

common_overrides:
  - experiment.name=sweep_arch_v1
  - wandb.group=sweep_arch_v1

variants:                        # 每个变体的特异化参数，可用内联 dict
  - name: mlp
    overrides:
      - model.name=mlp
      - "model.config={hidden_dims: [128, 64], dropout: 0.2, use_norm: true}"
  - name: lstm
    overrides:
      - model.name=lstm
      - "model.config={hidden_dim: 64, num_layers: 2, dropout: 0.2, pooling: last}"

grid:                            # 笛卡尔展开，叠加在每个 variant 上
  train.lr: [0.001, 0.0005]
  experiment.seed: [42, 123]

slurm:                           # 仅 --launcher slurm 时使用
  extra_sbatch: []               # 追加 #SBATCH 行，如 ["--account=prjs1859", "--qos=normal"]
  # env_script: "${ROOT_BASE}/load_exp_env.sh"   # 每份 spec 可指定环境激活脚本；
  #                                              # 资源申请写在 scripts/slurm/train.sbatch.tmpl 头部
```

上例 = 2 variants × 2 lr × 2 seed = **8 个 job**。

### 跨环境：一份 sweep spec 对应一个环境

`sweep.py` 不切 Python 解释器，`entry: train` 的每个 job 是单模型、后端缺失就响亮失败。
所以 **TSFM 系列按环境各写一份 sweep spec**，每份只列该环境能跑的模型：

- `experiments/sweep.example.yaml` —— 纯 Python 架构（mlp / lstm / dlinear），任何环境都能跑。
- `experiments/sweep.chronos.yaml` —— `chronos_bolt_head_only` 等，需 `chronos` 后端，从装了它的解释器里跑。
- 要扫别环境的模型（如 3.9–3.11 环境的 `moment`），拷一份、换掉 variants、从那个解释器里跑。

用法就是「手动切到对应环境 → `python scripts/sweep.py -s <该环境的 spec>`」，下游 sbatch 生成与提交无感知。
`--launcher local` / `print` 用当前 PATH 上的 `python` 即可；`--launcher slurm` 时每份 spec 可用
`slurm.env_script` 指定 `.sbatch` 里 `source` 的环境激活脚本（不设则沿用
`${ROOT_BASE}/load_exp_env.sh`）。

```bash
python scripts/sweep.py -s experiments/sweep.example.yaml --launcher print               # 只打印命令
python scripts/sweep.py -s experiments/sweep.example.yaml --launcher local --max-parallel 2
python scripts/sweep.py -s experiments/sweep.example.yaml --launcher slurm                # 渲染模板并 sbatch
python scripts/sweep.py -s experiments/sweep.example.yaml --launcher slurm --dry-run      # 只写 .sbatch 不提交
```

| launcher | 行为 |
| --- | --- |
| `print` | 每个 job 打印一行 `python scripts/<entry>.py -c <该 job 的 config>` |
| `local` | 用线程池并发跑子进程，`--max-parallel N`，各 job 日志写 `logs/<job>.log`，末尾汇总成败 |
| `slurm` | 逐 job 用 `scripts/slurm/train.sbatch.tmpl` 渲染出 `.sbatch` 并 `sbatch` 提交（`--dry-run` 只写不提交） |

### `--launcher slurm` 的产物与手动提交

一次 `slurm` 调度（含 `--dry-run`）在 `runs/_sweeps/<时间戳>_<spec 名>/` 下留两份东西：

```text
runs/_sweeps/20260831-150217_sweep.example/
├── configs/  <job>.yaml   ×N     # 每个 job 的完整解析配置（核心账本）
└── sbatch/   <job>.sbatch ×N     # 由 train.sbatch.tmpl 渲染，可直接 sbatch
```

`train.sbatch.tmpl` 现在承载真实集群头：`#SBATCH -p gpu_h100 / --gpus=1 / --mem-per-gpu=...`、
`ROOT_BASE` 双挂载解析（`/projects` 与 `/gpfs`）、`source "${ROOT_BASE}/load_exp_env.sh"` 激活环境、
torch 自检。sweep 只替换 5 个 `@@...@@` 标记（`JOB_NAME` / `PROJECT_DIR` / `ENTRY_SCRIPT` /
`CONFIG_PATH` / `EXTRA_SBATCH`），bash 体原样保留。渲染出的每个 `.sbatch` 里那句业务就是：

```bash
python scripts/train.py -c runs/_sweeps/<...>/configs/<job>.yaml
```

所以 `--launcher slurm`（不加 `--dry-run`）会直接提交全部；若想审阅后再投，用 `--dry-run` 生成，
然后手动挑着提交：

```bash
cd runs/_sweeps/20260831-150217_sweep.example
sbatch sbatch/lstm__lr-0.001__seed-42.sbatch      # 单个
for f in sbatch/lstm__*.sbatch; do sbatch "$f"; done   # 一批
```

每个 job 的训练结果仍按 `run.name` 落在 `runs/<experiment.name>/<job>/`（config / checkpoints /
metrics.json …），和手动跑一次 `scripts/train.py` 完全一致。

投之前按自己账号改 `train.sbatch.tmpl` 头部的 `-p` / `-t` / `--mem-per-gpu` / `-o` / `-e`。
默认 `export WANDB_MODE=online`；算力节点若无外网，用 `WANDB_MODE=offline sbatch ...`
覆盖，跑完在登录节点 `wandb sync runs/**/wandb/offline-run-*` 回传。

`scripts/run_all_models.sh [config.yaml] [extra.override=value ...]` 是不需要 sweep 规格、
本地顺序跑完 `mlp / lstm / dlinear` 的便捷脚本。

---

## 6. Run 目录布局

```text
runs/<experiment.name>/<run.name>/
├── config.yaml               # 解析后的完整配置（可原样 -c 重投）
├── checkpoints/
│   └── <modelclass>_best.pt  # 按 val rank IC 选出的最佳权重
├── history.json              # 逐 epoch 的 train/val loss、rank IC、lr
├── metrics.json              # OOS 测试 + 回测汇总
├── test_predictions.csv      # 测试集逐样本预测（train 流程）
├── predictions.csv           # 推理输出（infer 流程，名字可由 --out 改）
└── reports/                  # 横评报告（benchmark 流程）
    └── benchmark_summary.{csv,md,json}
```

---

## 7. wandb 接入

`wandb` 为**硬依赖**。关闭某次 run：`wandb.enabled=false` 或 `wandb.mode=disabled`
（此时 `WandbTracker` 全部方法为 no-op，不建 run、不联网）。

`wandb.mode` 默认 `online` 时不会显式传给 `wandb.init`，因此环境变量 `WANDB_MODE` 生效
（`train.sbatch.tmpl` 在算力节点默认 `export WANDB_MODE=offline`，跑完在登录节点
`wandb sync runs/**/wandb/offline-run-*` 回传）；配置里显式写 `wandb.mode=offline` 也可强制。

| 时机 | 上报内容 |
| --- | --- |
| 每 `wandb.log_freq` 个 epoch | `train/loss`, `val/loss`, `val/mean_rank_ic`, `val/ic_ir`, `lr`, `epoch`（经 `Trainer(callbacks=[...])` 钩子） |
| 训练结束 | OOS + 回测指标写入 wandb summary，best checkpoint 作为文件上传 |
| 横评 | 汇总表作为 `wandb.Table`，最优行写 summary |

run 归组：`wandb.group`（默认 `experiment.name`）、`wandb.job_type`（`train` / `infer` / `benchmark`）、
`wandb.name`（默认 run 文件夹名）。

---

## 8. 编程接口

脚本只是薄壳，同样的流程可直接在 Python 里调：

```python
from crossec_forecast.pipelines import load_experiment, run_train

cfg, run_dir, logger = load_experiment(
    "experiments/experiment.yaml",
    ["model.name=lstm", "train.epochs=30", "wandb.enabled=false"],
    job_type="train",
)
result = run_train(cfg, run_dir, logger)
print(result["summary"], result["checkpoint"])
```

其它可用入口：`load_config`（只合并不建 run）、`run_infer`、`run_benchmark`、
`to_data_config` / `to_train_config`（schema → 库内 dataclass）、`WandbTracker`。

---

## 9. 注意事项

- **浮点覆盖要带小数点**：`train.lr=0.0005`，不要写 `5e-4`（YAML 会当字符串）。
- **struct 模式**：schema 未定义的 key 会报错，是有意的拼写保护；要加新配置项先改 `experiment_schema.py`。
- **`feature_cols` vs `feature_pattern`**：`data.feature_cols` 给显式列表时**覆盖** `feature_pattern`；
  两者都会保证 train/val/test 三份切分用同一批特征列。
- **推理输出为 CSV**：不额外依赖 parquet 引擎；需要 parquet 自行在 `run_infer` 结果上转。
- **SLURM 路径**：`.sbatch` 在“执行 `sweep.py` 的机器”上渲染，路径风格随该 OS；在 Linux 集群上生成即为 POSIX 路径。
- **产物默认 gitignore**：`runs/`、`wandb/`、`*.pt`、`benchmark_reports/` 均已忽略。
- **TSFM 后端按 Python 版本分环境**：部分库依赖冲突不能共存（如 `chronos-forecasting` 要较新 Python，
  `momentfm` 的旧 `transformers` pin 要 Python ≤3.11）。按**推荐 Python 版本**分别建环境（环境名自定）：
  Python 3.11+ 装 `.[chronos]`，Python 3.9–3.11 装 `.[moment]`。wrapper 用
  `models._optional.require_modules` 惰性 import + `REQUIRED_MODULES` / `PYTHON_HINT` 声明；后端缺失的
  模型仍会注册但 `build_model` 报 `ModelDependencyError`，`BenchmarkEngine` 扫描整份名单、自动跳过
  当前环境跑不了的并告警。共用一份 `benchmark.models` 名单，各环境跑各自能跑的子集，跑完 concat 各份
  `benchmark_summary.csv` 再按 `test_rank_ic` 排序即可。详见
  [`crossec_forecast/models/pretrained_research.md`](./crossec_forecast/models/pretrained_research.md) §2。
