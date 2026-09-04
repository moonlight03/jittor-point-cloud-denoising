# 提交代码说明

本代码包包含 OBJ 数据处理、随机初始化训练、held-out 验证与最佳模型选择、单卡测试集推理、结果校验和打包。

## 代码结构

```text
code/
  main.py                 统一命令入口
  prepare_data.py         OBJ 到 NPY 的确定性数据准备
  data.py                 数据读取、归一化与训练 patch 采样
  model.py                三阶段 heavy 去噪模型
  emd.py                  自定义 CUDA Auction EMD
  train.py                随机初始化训练、续训、checkpoint
  validate.py             held-out 推理、CD/P2S、best 选择
  denoise.py              单卡 patch 推理与 stitching
  datalist/               训练/验证/测试 ID 清单
  scripts/                数据、训练和推理单卡脚本
requirements.txt          pip 依赖及固定版本
提交说明文档.pdf           官方要求的中文说明文档
```

### 文件功能说明

| 文件 | 功能 |
|---|---|
| `main.py` | argparse 统一入口，提供 7 个子命令：`prepare-data`、`verify-test`、`train`、`validate`、`infer`、`validate-predictions`、`package-predictions` |
| `prepare_data.py` | 读取官方 ShapeNet OBJ，按三角形面积采样点云，单位球归一化，确定性生成训练 NPY 和验证 NPY（含 Laplace 噪声） |
| `data.py` | `PairedPatchDataset` 负责训练时的 clean/noisy patch 成对采样，KNN 切 patch，在线加噪 |
| `model.py` | `DenoiseFlow` 三阶段串联模型，每阶段含 10 层 EdgeConv + 10 个可逆单调流（FlowAssembly），谱范数约束 |
| `emd.py` | Auction 算法的 CUDA 实现，固定 1024 点、eps=0.005、50 次迭代，通过 `jt.code` 即时编译 |
| `train.py` | 随机初始化训练，损失为 EMD+CD+Repulsion，plateau 学习率，每 5 epoch 做 held-out 验证并保存 best |
| `validate.py` | 对 checkpoint 在 200 项 held-out 上推理，计算 CD/P2S/Final，只有完整验证且 Final 严格更高时更新 best |
| `denoise.py` | 单卡推理：单位球归一化 → FPS 选 patch 中心 → KNN 切 patch → 模型去噪 → poly 加权拼回整云；含预测校验和打包 |
| `scripts/env.sh` | 设置 PYTHONPATH、JITTOR_HOME、编译器路径 |
| `scripts/prepare_data.sh` | 数据准备封装脚本 |
| `scripts/train_single_gpu.sh` | 单卡训练封装脚本 |
| `scripts/infer_single_gpu.sh` | 单卡推理封装脚本（含 verify-test → infer → validate-predictions → package-predictions） |
| `scripts/full_pipeline_three_commands.sh` | 完整三步流水线：数据准备 → 训练 → 推理 |

### 核心模块逻辑

#### 数据准备（prepare_data.py）

读取 `dataset_train/shapenet/<synset>/<model>/models/model_normalized.obj`，对每个 mesh 按三角形面积采样点云，用 bbox 中心平移、最远半径缩放做单位球归一化。每个 shape 的随机种子由 `全局 seed + shape ID + 数据流名称` 的 SHA256 派生，并行 worker 数量不影响结果。

- 训练集：15,633 个 `float32 [50000, 3]` NPY
- 验证集：200 组，每组含 `clean.npy`（400k 点）、`clean_50k.npy`（50k 点）和带确定性 Laplace 噪声的 `noisy.npy`（50k 点，噪声强度 uniform[0.004, 0.017]）
- manifest 记录每个输入/输出的 seed、shape、dtype、SHA256

#### 模型（model.py）

`DenoiseFlow` 是单个去噪阶段，heavy 配置下三个阶段串联：

1. **特征提取**：patch 内 KNN 图（k=32），PreConv 编码局部几何，10 层 EdgeConv 逐层聚合邻域特征
2. **噪声通道**：NoiseEdgeConv 根据坐标邻域估计 32 维附加噪声通道
3. **可逆单调流**：10 个 FlowAssembly，每个含 2 个 IMonotoneBlock + 2 个 ActNorm，通过 12 次不动点迭代求解隐变量
4. **去噪**：进入潜空间后将最后 16 维噪声通道置零，再通过逆变换恢复三维坐标
5. **谱约束**：InducedNormLinear 用幂迭代约束 Lipschitz 常数 ≤ 0.98，保证不动点求逆收敛

三个阶段的输出均参与 EMD 损失（stage_weights = 1:1:1），最终阶段输出额外计算 CD 和 Repulsion。

#### 训练（train.py）

随机初始化，不加载预训练模型。每个 batch 的训练流程：

1. 从 cached NPY 加载 clean 点云，单位球归一化
2. 在线加 Laplace 噪声（强度 uniform[0.004, 0.017]），随机缩放（0.8-1.2），可选旋转增强
3. KNN 切 1024 点 patch，clean/noisy 取相同索引保持逐点配对
4. 三阶段模型前向，得到 3 个阶段的预测输出
5. 损失计算：`loss = 0.00625 * EMD + 40.0 * CD + 40.0 * Repulsion`
   - EMD：Auction 算法在 1024 点上求一一匹配，三阶段输出分别计算后求和
   - CD：Chamfer Distance，只在最终阶段输出上计算
   - Repulsion：k=8 最近邻排斥，半径 r0=0.05，防止点聚集
6. 梯度裁剪（max_norm=0.001），Adam 优化器，plateau 学习率调度
7. 每 5 个完整 epoch 做 held-out 验证，只有完整验证的 Final 严格更高时才更新 `best_val_model.pkl`

#### 验证与 best 选择（validate.py）

每 5 个 epoch 在 200 项 held-out 上推理（patch_size=2048, seed_k=4, niters=2, stitch=poly），计算每个样本的：

- CD：预测 vs clean 的双向 Chamfer Distance
- P2S：预测点到 mesh 曲面的距离（用 clean 参考坐标归一化）
- Final：`0.5 * cd_score + 0.5 * p2s_score`，其中 score = `100 * (1 - pred_error / noisy_error)`

只有 `complete=True`（200 项全部完成）且 Final 严格高于当前 best 时才更新：

```text
checkpoints/best_val_model.pkl
checkpoints/best_val_state.pkl
best_val_summary.json
```

#### 推理（denoise.py）

单卡处理 200 项测试输入，每个点云的推理流程：

1. 单位球归一化
2. FPS 选择 patch 中心（seed_k=4，约 97 个 patch）
3. KNN 切 2048 点 patch，减去 seed 中心坐标
4. 两轮 patch 去噪（niters=2），每轮分批次前向（patch_batch=12）
5. poly 加权拼回整云：按离 patch 中心的距离 `max(0, 1-d)^1.0` 加权融合
6. 反归一化，`fix_count` 保证输出点数与输入一致
7. 保存为 `denoised.npy`（float32）

## 环境配置

### 系统要求

- 操作系统：Ubuntu 22.04
- CUDA：12.4（兼容 12.x）
- Python：3.9
- GPU：NVIDIA 4090（或其他 CUDA 兼容 GPU）

### 安装步骤

```bash
python -m pip install -r requirements.txt
```

验证环境可用：

```bash
python -c 'import jittor as jt; jt.flags.use_cuda=1; print(jt.__version__)'
```

`scripts/env.sh` 会在运行 shell 脚本时自动设置 PYTHONPATH、JITTOR_HOME 和编译器路径（g++/nvcc），无需手动 export。

## 运行步骤

### 完整三步流水线

```bash
bash code/scripts/full_pipeline_three_commands.sh \
  /path/to/dataset_train \
  /path/to/dataset_test_noisy \
  /path/to/full_reproduction_run \
  0
```

参数说明：
- `dataset_train`：官方训练 OBJ 根目录
- `dataset_test_noisy`：官方测试 noisy NPY 根目录
- `full_reproduction_run`：输出根目录
- `0`：GPU 索引

最终产出：
- 模型：`full_reproduction_run/train_rotation_none/checkpoints/best_val_model.pkl`
- 预测：`full_reproduction_run/test_best_val/predictions/shapenet/<synset>/<model>/denoised.npy`
- 提交：`full_reproduction_run/test_best_val/result.zip`

### 分步执行

#### 步骤 1：数据准备

官方训练 OBJ 结构：

```text
dataset_train/shapenet/<synset_id>/<model_id>/models/model_normalized.obj
```

```bash
bash code/scripts/prepare_data.sh \
  /path/to/dataset_train \
  /path/to/prepared_data \
  rotation_none_epoch98
```

产出：
- `prepared_data/train/*.npy`：15,633 个训练点云
- `prepared_data/heldout/shapenet/*/*/`：200 组验证数据（clean.npy、clean_50k.npy、noisy.npy）
- `prepared_data/manifest.json`：完整审计记录

#### 步骤 2：单卡训练

```bash
bash code/scripts/train_single_gpu.sh \
  /path/to/prepared_data/train \
  /path/to/prepared_data/heldout \
  /path/to/dataset_train \
  /path/to/train_run \
  0
```

参数说明：
- `TRAIN_DATA_ROOT`：训练 NPY 目录
- `VAL_ROOT`：held-out 验证目录
- `OBJ_MESH_ROOT`：训练 OBJ 根目录（验证时 P2S 需要 mesh）
- `RUN_ROOT`：训练输出目录
- `GPU_INDEX`：GPU 索引

产出：
- `train_run/checkpoints/best_val_model.pkl`：最佳模型权重
- `train_run/checkpoints/latest_state.pkl`：续训状态（含模型、Adam、epoch、best score）
- `train_run/best_val_summary.json`：最佳验证分数
- `train_run/validation/epoch_XXXX/`：各轮验证明细

显式续训：

```bash
RESUME_STATE=/path/to/train_run/checkpoints/latest_state.pkl \
  bash code/scripts/train_single_gpu.sh \
  /path/to/prepared_data/train \
  /path/to/prepared_data/heldout \
  /path/to/dataset_train \
  /path/to/resumed_run 0
```

#### 步骤 3：单卡测试集推理

推理前需通过 `CKPT` 环境变量指定权重。

官方测试输入结构：

```text
dataset_test_noisy/shapenet/<synset_id>/<model_id>/noisy.npy
```

```bash
CKPT=/path/to/best_val_model.pkl \
bash code/scripts/infer_single_gpu.sh \
  /path/to/dataset_test_noisy \
  /path/to/inference_run \
  0
```

脚本依次执行：verify-test（校验 200 项输入）→ infer（生成 denoised.npy）→ validate-predictions（校验输出格式）→ package-predictions（打包 result.zip）。

产出：
- `inference_run/predictions/shapenet/<synset>/<model>/denoised.npy`
- `inference_run/result.zip`：A 榜提交文件

### 关键超参数

| 参数 | 默认值 | 说明 |
|---|---|---|
| `--max-epochs` | 100 | 最大训练轮数 |
| `--batch-size` | 16 | 每 batch 的 patch 数 |
| `--train-patch-size` | 1024 | 训练 patch 点数 |
| `--lr` | 2e-4 | 初始学习率 |
| `--min-lr` | 1e-6 | plateau 衰减下限 |
| `--lr-schedule` | plateau | 学习率调度策略 |
| `--plateau-patience` | 2 | plateau 容忍轮数 |
| `--plateau-factor` | 0.5 | plateau 衰减系数 |
| `--emd-weight` | 0.00625 | EMD 损失权重 |
| `--cd-weight` | 40.0 | CD 损失权重 |
| `--repulsion-weight` | 40.0 | Repulsion 损失权重 |
| `--repulsion-radius` | 0.05 | 排斥半径 r0 |
| `--repulsion-k` | 8 | 排斥最近邻数 |
| `--grad-clip` | 0.001 | 梯度裁剪 max_norm |
| `--noise-min` | 0.004 | 在线噪声下限 |
| `--noise-max` | 0.017 | 在线噪声上限 |
| `--validate-every` | 5 | 验证间隔（epoch） |
| `--save-every` | 5 | checkpoint 保存间隔（epoch） |
| `--early-stop-patience` | 15 | 早停容忍轮数 |
| `--early-stop-min-epoch` | 45 | 早停最早触发轮数 |
| `--seed` | 2023 | 全局随机种子 |

推理参数（`infer` 命令）：

| 参数 | 默认值 | 说明 |
|---|---|---|
| `--patch-size` | 2048 | 推理 patch 点数 |
| `--seed-k` | 4 | FPS 中心数 = seed_k * N / patch_size |
| `--niters` | 2 | 去噪迭代轮数 |
| `--patch-batch` | 12 | 单次前向的 patch 数 |
| `--stitch` | poly | patch 拼接模式 |
| `--stitch-power` | 1.0 | poly 加权指数 |
| `--rotation` | identity | 推理旋转模式 |
| `--seed` | 2023 | 推理随机种子 |

### A 榜提交格式

训练完成后，`result.zip` 即为 A 榜提交文件，内部结构为：

```text
result.zip
  shapenet/<synset_id>/<model_id>/denoised.npy
```

每个 `denoised.npy` 为 `float32 [N, 3]`，N 与对应输入 `noisy.npy` 的点数一致。`validate-predictions` 命令会校验 200 个 ID 齐全、形状匹配、dtype 为 float32 且无 NaN/Inf。
