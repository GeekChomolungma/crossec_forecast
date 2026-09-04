# 配置覆盖与三条 Pipeline 消费手册

> 面向"这个配置到底怎么流转、这条 pipeline 最后拿到的是什么"这个问题。
> 概念背景（`experiment_schema.py` vs `default_config.py`、YAML 结构总览）见
> [`EXPERIMENTS.md`](../EXPERIMENTS.md);本文档只讲**覆盖链的精确规则**和
> **三条 pipeline 各自最终生效的配置字段**。

---

## 0. 一句话结论

```
schema 默认值  <  -c 文件1  <  -c 文件2  < ...  <  CLI KEY=VALUE
```

右边永远覆盖左边。**dict 按 key 递归合并,list 整体替换**(重要,见 §3)。三条 pipeline
（`train` / `infer` / `benchmark`）从同一份合并结果 `cfg` 里**各取各的段**,互不干扰——
`model:` 只对 train/infer 有意义,`benchmark.models[]` 只对 benchmark 有意义,这不是巧合,
是三条 pipeline 从设计上就读不同的字段（见 §2 的"谁读谁"总表）。

---

## 1. 覆盖链的四层来源

| 层级 | 来源 | 代码位置 | 特点 |
|---|---|---|---|
| ① schema 默认值 | `ExperimentConfig`(及其嵌套 `DataSchema`/`TrainSchema`/`ModelSchema`/`BenchmarkSchema`/`WandbSchema`) | [`configs/experiment_schema.py`](../crossec_forecast/configs/experiment_schema.py) | 永远是合并的**最底层**;`OmegaConf.structured(...)` 生成,进入 **struct 模式**——之后任何一层写了 schema 里没有的 key 都会报错(拼写保护) |
| ② 基底 YAML | `experiments/experiment.yaml` | 手写文件 | "规范全量配置",团队共享的默认实验设定 |
| ③ 一个或多个 `-c` 片段 | `experiments/models/*.yaml` 等 | 手写文件,`-c` **可重复**,按命令行给出的顺序**从左到右**依次合并 | 只写自己要覆盖的 key,其余继承前面所有层 |
| ④ CLI dot-list 覆盖 | 命令行位置参数 `KEY=VALUE` | `OmegaConf.from_dotlist(overrides)` | 优先级最高,常用于一次性微调(`train.lr=0.0005`) |

合并本身只有一行代码([`pipelines/context.py::load_config`](../crossec_forecast/pipelines/context.py)):

```python
layers = [OmegaConf.structured(ExperimentConfig)]     # ①
for p in paths: layers.append(OmegaConf.load(str(p))) # ②③(paths = [experiment.yaml, 片段1, 片段2, ...])
layers.append(OmegaConf.from_dotlist(overrides))       # ④
cfg = OmegaConf.merge(*layers)
```

`scripts/train.py` / `infer.py` / `benchmark.py` 三个入口的 `-c`(`action="append"`)和位置参数
`KEY=VALUE` 就是在给这个函数拼 `paths` 和 `overrides`,三者共用同一套解析逻辑
([`scripts/_common.py`](../scripts/_common.py))。

**开放字典例外**:`model.config` 是 `Dict[str, Any]`,**没有 struct 保护**,里面随便写键值都不报错——
模型特异参数(`hidden_dim`、`loss_type`、`sign`、…)都塞在这里,不用改 schema。

**可复现**:一次 `load_experiment` 会把①~④合并、解析(`${...}` 插值展开)后的**完整快照**落盘到
`runs/<experiment.name>/<run.name>/config.yaml`,以后可以原样当作单独一个 `-c` 重投,等价于当时那一整条覆盖链。

---

## 2. 谁读谁——三条 pipeline 消费矩阵

**这是最容易搞错的地方,先看这张表。** ✓ = 这个字段会真正影响该 pipeline 的行为;✗ = 合并进
`cfg` 了,但该 pipeline 的代码根本不会去读它,写了也白写。

| 顶层段 / 字段 | `run_train`<br>(scripts/train.py) | `run_infer`<br>(scripts/infer.py) | `run_benchmark`<br>(scripts/benchmark.py) |
|---|:---:|:---:|:---:|
| `experiment.name` | ✓(run 目录 / wandb group) | ✓(run 目录) | ✓(run 目录 / wandb group) |
| `experiment.seed` | ✓(全局一次) | ✓(全局一次,推理无随机性,基本无感) | ✓(全局一次 + **每个模型开跑前重新 seed 一次**) |
| `run.name` / `output_root` | ✓ | ✓ | ✓ |
| `run.tags` | ✓(wandb tags) | ✗(infer 不接 wandb) | ✓(wandb tags) |
| `data.*`(`path`/列选取/`seq_len`/`batch_size`/`num_workers`) | ✓ | ✓(**不含** `split.*`、`shuffle_train`、`drop_last`——推理不切分、不 shuffle) | ✓(和 train 一样走 `build_dataloaders`,全量生效) |
| `data.split.*` | ✓ | ✗ | ✓ |
| `train.*`(epochs/lr/调度器/早停…) | ✓ | 只用 `train.device` 选设备,其余全部 ✗ | ✓(**所有 roster 模型共用同一份**,保证横评公平) |
| `model.name` / `model.config` | ✓(唯一入口) | ✓(唯一入口) | **✗✗✗ 完全不读**(见下方"重要陷阱") |
| `benchmark.top_quantile` | ✓(只用来算 OOS 测试和回测,不涉及 roster) | ✗ | ✓ |
| `benchmark.models[]` | ✗ | ✗ | ✓(唯一入口) |
| `wandb.*` | ✓ | ✗(infer 目前不接入 `WandbTracker`) | ✓ |

### 重要陷阱:`model:` 块对 benchmark pipeline 是死代码

[`benchmark_pipeline.py::run_benchmark`](../crossec_forecast/pipelines/benchmark_pipeline.py) 全程
**没有任何一行读 `cfg.model`**。真正决定跑哪些模型的只有 `cfg.benchmark.models[]`。

那为什么很多模型片段(如 [`lstm.yaml`](models/lstm.yaml))还是只写了 `model:`?——因为那些模型不需要
覆盖 `data:`,天然可以待在 `experiment.yaml` 自带的共享 `benchmark.models` 列表里,**不需要**自己的片段
再单独定义一份 roster;`model:` 块留着只是为了同一份片段也能给 `train`/`infer` 用。

但**有一个隐藏的副作用**:[`resolve_run_dir`](../crossec_forecast/pipelines/context.py) 在自动生成
run 文件夹名时,不管当前是哪条 pipeline,统一读的是 `cfg.model.name`:

```python
run_name = f"{ts}_{cfg.model.name}_{config_hash(cfg)}"
```

所以一次 benchmark 跑了 `[mlp, lstm, dlinear]` 三个模型,如果 `run.name` 没手动指定,文件夹名里的那个
"模型名"其实是 `cfg.model.name`(schema 默认 `"mlp"`,或者哪个片段最后设置的值),和 roster 里实际跑的
模型没有必然关系——这也是为什么像 [`chronos_bolt_zeroshot.yaml`](models/chronos_bolt_zeroshot.yaml)
这种片段即使只服务 benchmark,也顺手把 `model.name` 设成同名,让 run 文件夹名不至于文不对题。

---

## 3. list 覆盖 = 整体替换,不是逐元素合并

OmegaConf 合并两层配置时:

- 两边都是 **dict**(`data:`、`train:`、`model.config` 内部…)→ 按 key **递归合并**,后面的层
  只覆盖它显式写出的 key,没写的 key 继续沿用前面层的值。
- 同一个 key 的值在后面的层里是 **list**(比如 `benchmark.models`)→ 后面这层的**整个 list 直接
  顶替**前面层的整个 list,不会按下标合并、也不会去重追加。

所以 [`chronos_bolt_zeroshot.yaml`](models/chronos_bolt_zeroshot.yaml) 里写
`benchmark.models: [仅自己一条]`,合并到 `experiment.yaml` 那份 `[moment_zeroshot, ...]` 上时,
**结果是整份替换成那一条**,不是变成两条列表拼接。这正是"给需要覆盖 `data:` 的模型单独开一份
只含自己的 benchmark roster"这个技巧能生效的根本原因。

---

## 4. 三条 pipeline 的"最终生效配置"详解

以下每条都给出:入口命令、内部读取链、以及一张"这条 pipeline 到底要哪些字段"的清单。

### 4.1 Train —— `scripts/train.py` → `run_train`

```bash
python scripts/train.py -c experiments/experiment.yaml
python scripts/train.py -c experiments/experiment.yaml -c experiments/models/lstm.yaml
python scripts/train.py -c experiments/experiment.yaml -c experiments/models/lstm.yaml train.lr=0.0005
```

```
cfg.data.*  ──to_data_config──▶ DataConfig ──build_dataloaders──▶ train/val/test DataLoader + meta
cfg.model.{name,config} + meta(seq_len,num_features,num_cov) ──model_build_config──▶ build_model(...)
cfg.train.* ──to_train_config(cfg, run_dir/checkpoints)──▶ TrainConfig ──▶ Trainer
cfg.benchmark.top_quantile ──▶ trainer.evaluate(...) / SimpleLongShortBacktester(...)  (仅这一个 benchmark 字段)
cfg.wandb.* + cfg.run.tags + cfg.experiment.name ──▶ WandbTracker
```

| 需要的段 | 用在哪 |
|---|---|
| `experiment.name`, `experiment.seed` | run 目录归组、全局随机种子 |
| `run.name`, `run.output_root`, `run.tags` | 产物路径、wandb tags |
| `data.*`(全部字段) | 数据切分、滑窗、特征/协变量列选取、DataLoader |
| `train.*`(全部字段) | 优化器、调度器、早停、梯度裁剪、设备 |
| `model.name`, `model.config` | **唯一决定训练哪个模型** |
| `benchmark.top_quantile` | OOS 测试与回测的多空分位(仅此一项) |
| `wandb.*` | 实验追踪 |
| ~~`benchmark.models[]`~~ | **不读,写了也不影响 train** |

### 4.2 Infer —— `scripts/infer.py` → `run_infer`

```bash
python scripts/infer.py -c runs/baseline_v1/<run>/config.yaml \
    --checkpoint runs/baseline_v1/<run>/checkpoints/lstmclassifier_best.pt

python scripts/infer.py -c experiments/experiment.yaml -c experiments/models/lstm.yaml \
    --checkpoint path/to/lstm_best.pt --data ./data/new_panel.csv --out preds_2026Q3.csv
```

`--checkpoint` / `--data` / `--out` 是**脚本自己的参数**,不在 YAML 体系里(`--data` 会覆盖
`cfg.data.path`,但只在这次调用生效,不会写回配置文件)。

```
cfg.data.*(部分字段) ──▶ PanelTimeSeriesDataset(is_inference=True, 不切分、全量打分)
cfg.model.{name,config} + dataset.(seq_len,num_features,num_cov) ──model_build_config──▶ build_model(...)
--checkpoint ──▶ model.load_state_dict(strict)
cfg.train.device ──▶ select_device(...)  (train.* 唯一被用到的字段)
```

| 需要的段 | 用在哪 | 备注 |
|---|---|---|
| `data.path` | 待打分的 panel(可被 `--data` 覆盖) | |
| `data.target_col`, `fwd_ret_col`, `timestamp_col`, `symbol_col` | 构造 `PanelTimeSeriesDataset` | 推理时 `target_col` 不参与 loss,但仍需列存在或走 `is_inference` 分支 |
| `data.feature_pattern`, `feature_cols`, `cov_pattern`, `cov_cols` | 决定打包进 `x` 的列 | 必须和训练该 checkpoint 时的设置**完全一致**,否则维度对不上 |
| `data.seq_len`, `batch_size`, `num_workers` | 滑窗长度、推理 batch | |
| `model.name`, `model.config` | 决定 `build_model` 出的模型结构 | 必须和 checkpoint 的模型结构一致 |
| `train.device` | 选 cpu/cuda/mps | `train.*` 里唯一用到的字段 |
| `--checkpoint`(CLI 独有) | 加载哪份权重 | 必填,不在 YAML 里 |
| ~~`data.split.*` / `shuffle_train` / `drop_last`~~ | 不读 | 推理不切分、不 shuffle |
| ~~`train.epochs`/`lr`/调度器/早停~~ | 不读 | 不训练 |
| ~~`benchmark.*`~~ | 不读 | |
| ~~`wandb.*`~~ | 不读 | infer 目前没有接入 `WandbTracker` |

### 4.3 Benchmark —— `scripts/benchmark.py` → `run_benchmark`

```bash
python scripts/benchmark.py -c experiments/experiment.yaml
python scripts/benchmark.py -c experiments/experiment.yaml -c experiments/models/chronos_bolt_zeroshot.yaml
```

```
cfg.data.* ──to_data_config──▶ DataConfig ──build_dataloaders──▶ 一份共享的 train/val/test DataLoader
cfg.benchmark.models[] ──▶ 逐个 {name, config} 循环 build_model(...)（跳过后端不可用的条目）
cfg.train.* ──to_train_config──▶ 同一份 TrainConfig，喂给每个模型各自的 Trainer(保证同条件横评)
cfg.benchmark.top_quantile ──▶ 每个模型的 OOS 评估 + 回测
cfg.experiment.seed ──▶ 全局种子 + BenchmarkEngine 每个模型开跑前重新 seed_everything
cfg.wandb.* ──▶ WandbTracker(汇总表 + 最优行写 summary)
```

| 需要的段 | 用在哪 |
|---|---|
| `experiment.name`, `experiment.seed` | run 目录归组;每个模型开跑前重新播种,保证各模型初始化条件一致 |
| `run.*` | 产物路径、wandb |
| `data.*`(全部字段,**只有一份,全 roster 共用**) | 保证"同一切分下"横评成立 |
| `train.*`(全部字段,**只有一份,全 roster 共用**) | 保证"同样超参下"横评成立 |
| `benchmark.models[]` | **唯一决定跑哪些模型**,每项 `{name, config}` |
| `benchmark.top_quantile` | 多空分位,应用到每个模型 |
| `wandb.*` | 汇总表 + 最优行 |
| ~~`model.name` / `model.config`~~ | **完全不读**(只是顺带被 `resolve_run_dir` 拿去生成文件夹名,见 §2) |

> 环境变量 `CF_BENCH_RUN_TEST=1`(不在 YAML 里,读的是进程环境变量)控制是否跑 OOS 测试 + 回测;
> 默认关闭,只跑 val,详见 [`eval/benchmark.py`](../crossec_forecast/eval/benchmark.py)。

---

## 5. 一个模型片段如何"一稿三用"

以 [`experiments/models/chronos_bolt_zeroshot.yaml`](models/chronos_bolt_zeroshot.yaml) 为例——
它需要覆盖 `data.feature_cols`(整跑生效,train/infer/benchmark 都吃到),同时用 YAML anchor 让
`model.config` 和 `benchmark.models[0].config` 保持一份定义、两处复用:

```yaml
data:
  feature_cols: [crossec_logret_1_mad_Zscore]     # 三条 pipeline 都吃这个覆盖

model:                                             # train / infer 用
  name: chronos_bolt_zeroshot
  config: &chronos_bolt_zeroshot_config
    model_id: amazon/chronos-bolt-small
    ...

benchmark:                                         # benchmark 用；自带单条目 roster，
  models:                                          # 因为它没法和共享 roster 共用同一份 data:
    - name: chronos_bolt_zeroshot
      config: *chronos_bolt_zeroshot_config
```

三条命令,三条 pipeline,同一份片段:

```bash
python scripts/train.py     -c experiments/experiment.yaml -c experiments/models/chronos_bolt_zeroshot.yaml
python scripts/infer.py     -c experiments/experiment.yaml -c experiments/models/chronos_bolt_zeroshot.yaml --checkpoint ...
python scripts/benchmark.py -c experiments/experiment.yaml -c experiments/models/chronos_bolt_zeroshot.yaml
```

- `train` / `infer` 走 `model:` 那一份;`benchmark` 走 `benchmark.models[0]` 那一份;`data.feature_cols`
  三条路都生效。
- SLURM 上等价地用 `scripts/slurm/benchmark.sbatch` 的 `CONFIG_EXTRA` 旋钮追加这份片段:
  `CONFIG_EXTRA="experiments/models/chronos_bolt_zeroshot.yaml" sbatch scripts/slurm/benchmark.sbatch`。

---

## 6. 三条 pipeline 的启动方式一览

三条 pipeline 各有一个"裸命令"入口(§4 已给出),但真正跑实验时还有三种更省事的方式:
本地顺序脚本、`sweep.py` 批量调度器、直接手动 `sbatch`。下表先给结论,再逐个展开。

| 场景 | 用什么 | 需要 sweep spec? | 走哪个/哪些脚本 |
| --- | --- | --- | --- |
| 单次调试一个模型 | 裸命令(§4.1/4.2) | 否 | `scripts/train.py` / `scripts/infer.py` |
| 本地顺序跑 mlp/lstm/dlinear 三个 baseline | `scripts/run_all_models.sh` | 否 | `scripts/train.py`(循环 3 次) |
| 本地/集群批量扫参(多模型 × 多超参网格) | `scripts/sweep.py` | 是 | `scripts/train.py` 或 `scripts/benchmark.py`(spec 里 `entry:` 决定) |
| 集群上跑一次性的"全家桶"多模型横评 | 直接 `sbatch scripts/slurm/benchmark.sbatch` | 否 | `scripts/benchmark.py` |
| 推理 / 打分 | 裸命令(§4.2) | 否 | `scripts/infer.py`(**目前没有 sbatch 封装**,见 §6.5) |

### 6.1 本地顺序跑 baseline —— `scripts/run_all_models.sh`

不需要任何 sweep spec,固定跑 `mlp / lstm / dlinear` 三个:

```bash
./scripts/run_all_models.sh                                   # 用 experiments/experiment.yaml
./scripts/run_all_models.sh experiments/experiment.yaml train.lr=0.0005 wandb.enabled=false
```

内部就是循环三次 `python scripts/train.py -c "$CONFIG" model.name=${M} run.name=quick_${M} ...`——
本质还是走 §4.1 的裸命令,只是省了手敲三遍。**不支持** `-c` 片段叠加(只有一个 `$CONFIG` 位置),
要用模型片段就直接用裸命令或 `sweep.py`。

### 6.2 批量调度器 —— `scripts/sweep.py`

一份 sweep spec(如 [`experiments/sweep.example.yaml`](sweep.example.yaml))声明:

```yaml
base_config: experiments/experiment.yaml   # 对应 §1 里的"基底 YAML"这一层
entry: train                               # train | benchmark —— 决定调用哪个入口脚本
common_overrides: [...]                    # 对应 CLI dot-list 这一层，所有 job 共享
variants: [{name: ..., overrides: [...]}]  # 每个变体的覆盖串，通常含 model.name / model.config
grid: {train.lr: [...], experiment.seed: [...]}  # 笛卡尔展开
```

`sweep.py` 对每个 `variant × grid点` 调一次 §1 的 `load_config(base_config, overrides)`,把
`common_overrides + variant.overrides + 该 grid 点`拼成 dot-list 覆盖,**在 sweep.py 自己这一步就把
配置完全解析、落盘成一份独立 YAML**(`runs/_sweeps/<戳>_<spec名>/configs/<job>.yaml`),再按
`--launcher` 派发:

```bash
python scripts/sweep.py -s experiments/sweep.example.yaml --launcher print                # 只打印命令，不执行
python scripts/sweep.py -s experiments/sweep.example.yaml --launcher local --max-parallel 2  # 本地线程池并发跑
python scripts/sweep.py -s experiments/sweep.example.yaml --launcher slurm [--dry-run]     # 渲染 .sbatch 并 sbatch 提交
```

`--launcher slurm` 渲染用的就是下面 §6.3 的模板。`entry: benchmark` 时,每个 job 仍然是"一个 job = 一份
完整 config",只是最终跑的是 `scripts/benchmark.py`——**注意**:`sweep.py` 不会展开
`benchmark.models[]` 本身,一个 job 的 `benchmark.models` 就是那份 config 里已经定好的 roster
(通常来自 `base_config` 或某个 `-c` 片段),`variants`/`grid` 展开的是 `model.*`/`train.*`/`data.*`
这类**单值**字段,不是拿它去扩增模型清单。

### 6.3 `scripts/slurm/train.sbatch.tmpl` —— `sweep.py --launcher slurm` 专用渲染模板

**不要直接 `sbatch` 这个文件**——它是给 `sweep.py` 当模板用的,单独提交没有意义(`@@TOKEN@@`
占位符不会被替换)。`sweep.py` 对每个 job 把 6 个 `@@...@@` 标记替换掉,渲染出
`runs/_sweeps/<戳>_<spec名>/sbatch/<job>.sbatch`:

| 标记 | 来自 |
|---|---|
| `@@JOB_NAME@@` | job 名(`variant名 + grid点`) |
| `@@PROJECT_DIR@@` | 仓库在集群上的路径 |
| `@@ENTRY_SCRIPT@@` | `scripts/train.py` 或 `scripts/benchmark.py`,取决于 spec 的 `entry:` |
| `@@CONFIG_PATH@@` | 该 job 已经完全解析落盘的那份 `configs/<job>.yaml` |
| `@@EXTRA_SBATCH@@` | spec 的 `slurm.extra_sbatch` 列表 |
| `@@ENV_SCRIPT@@` | spec 的 `slurm.env_script`,默认 `${ROOT_BASE}/load_exp_env.sh` |

**注意名字有点误导**:虽然文件名叫 `train.sbatch.tmpl`,但它其实是**通用单 job 启动模板**——
`@@ENTRY_SCRIPT@@` 既可以是 `train.py` 也可以是 `benchmark.py`,由 spec 的 `entry:` 字段决定,没有
一份专门叫 `benchmark.sbatch.tmpl` 的东西。渲染出的每个 `.sbatch` 业务行就是单独一句
`python <entry脚本> -c <这个 job 的完整配置>`——**没有 `-c` 拼接**,因为拼接这一步在 `sweep.py`
物化配置时已经做完了。

```bash
# --launcher slurm 不加 --dry-run 会直接全部提交；--dry-run 只生成 .sbatch，自己再挑着 sbatch
cd runs/_sweeps/20260904-xxxxxx_sweep.example
sbatch sbatch/lstm__lr-0.001__seed-42.sbatch
```

### 6.4 `scripts/slurm/benchmark.sbatch` —— 独立的一次性多模型横评作业

和 §6.3 完全不同的另一条路:**不经过 `sweep.py`**,是一份可以直接 `sbatch` 提交的完整脚本,自己在
脚本内部做 `-c` 拼接(对应 §1 的四层覆盖链里的②③层):

```bash
cd <集群上的仓库目录>

# 最简单：只用共享 experiment.yaml 里的 benchmark.models
sbatch scripts/slurm/benchmark.sbatch

# 叠加一个或多个模型片段（空格分隔，对应多个 -c），外加临时覆盖
CONFIG=experiments/experiment.yaml \
CONFIG_EXTRA="experiments/models/chronos_bolt_zeroshot.yaml" \
OVERRIDES="benchmark.top_quantile=0.1 wandb.job_type=benchmark" \
sbatch scripts/slurm/benchmark.sbatch
```

脚本内部拼出的命令等价于:

```bash
python scripts/benchmark.py -c experiments/experiment.yaml \
    -c experiments/models/chronos_bolt_zeroshot.yaml \
    benchmark.top_quantile=0.1 wandb.job_type=benchmark
```

四个环境变量旋钮:`CONFIG`(基底 YAML)、`CONFIG_EXTRA`(额外 `-c` 片段,空格分隔,可多个)、
`OVERRIDES`(CLI dot-list,空格分隔)、`PROJECT_DIR`(集群上的仓库路径)。解释器切换靠脚本里那一行
写死的 `source "${ROOT_BASE}/load_exp_env.sh"`,手动改路径切环境(想跑 `chronos` 后端的模型就指向
装了 `.[chronos]` 的环境)。

### 6.5 Infer 目前没有 SLURM 封装

`scripts/slurm/` 下只有 `train.sbatch.tmpl` 和 `benchmark.sbatch`,没有对应 `infer` 的模板。目前
推理只能走 §4.2 的裸命令(本地或手动登录计算节点跑),或者照抄 `benchmark.sbatch` 的结构自己写一份
——这是当前唯一还没补齐的一角,先记在这里。

---

## 7. 速查:常见坑

| 现象 | 原因 | 对策 |
|---|---|---|
| 把某模型塞进共享 `experiment.yaml` 的 `benchmark.models[]`,跑起来不报错但指标很奇怪 | 该模型需要专属 `data:` 覆盖(如 `feature_cols`/`target_col`),共享 roster 只有一份 `data:`,模型静默吃到了不对的输入维度 | 该模型自己的片段里单独定义一份只含自己的 `benchmark.models`(§5),别塞共享 roster |
| 改了 `model.config` 里的某个新超参,benchmark 跑出来没变化 | benchmark pipeline 根本不读 `cfg.model`(§2) | 改的是 `benchmark.models[].config`,不是 `model.config` |
| 覆盖 `benchmark.models` 后,发现共享 roster 里原来的模型"消失"了 | list 是整体替换,不是追加(§3) | 想追加就把原列表元素也抄一遍写全,或者干脆只用 CLI `benchmark.models=...` 一次性给完整列表 |
| YAML 里写了 schema 没有的顶层 key(如拼错 `trian.lr`) | struct 模式的拼写保护生效 | 检查 key 拼写;真要加新顶层配置项,先去改 [`experiment_schema.py`](../crossec_forecast/configs/experiment_schema.py) |
| `model.config` 里写了个奇怪的 key,没有任何报错 | `model.config` 是开放 `Dict[str, Any]`,没有 struct 保护 | 模型自己的 `__init__`/`config.get(...)` 里做校验(参考 `moment_zeroshot` 对 `anomaly_criterion` 的显式校验) |
| 浮点覆盖写了 `train.lr=5e-4` 没生效 | YAML/dotlist 会把它当字符串,不是浮点 | 写成带小数点的 `0.0005` |
| infer 时报维度不匹配 | `data.feature_cols`/`cov_cols`/`seq_len` 和训练该 checkpoint 时不一致 | infer 复用训练 run 落盘的 `runs/.../config.yaml` 作为 `-c`,而不是手写一份新的 |
