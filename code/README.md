# A+B 两阶段点云去噪（Jittor）

本目录包含 Heavy 模型的数据准备、两阶段训练、单卡/四卡推理和格式校验代码。
数据集不随目录分发，所有 OBJ、split 清单、NPY 和运行输出均通过命令行指定。

## 环境

已验证 Python 3.8/3.9、Jittor 1.3.11 和 CUDA 12.1/12.2；其余依赖版本见
`requirements.txt`。运行前激活任意满足依赖的 Jittor 环境：

```bash
conda activate your_jittor_env
python -c "import jittor; print(jittor.__version__)"
```

所有入口使用当前 shell 中的 `python`，不会覆盖解释器、编译器、CUDA 或动态库
路径。

## A/B OBJ 转 NPY

split 文件每行必须是 `shapenet/<类别>/<模型>`，对应 OBJ 路径为
`<MESH_ROOT>/<ID>/models/model_normalized.obj`。采样过程使用按样本派生的稳定
seed，按三角形面积均匀采样、单位球归一化，并保存为有限的
`(50000,3)`、`float32` NPY。

分别准备 A、B 数据：

```bash
WORKERS=8 SEED=2023 bash scripts/prepare_meshes.sh \
  a /path/to/a_meshes /path/to/a_npy /path/to/a_split.txt

WORKERS=8 SEED=2023 bash scripts/prepare_meshes.sh \
  b /path/to/b_meshes /path/to/b_npy /path/to/b_split.txt
```

可以在命令末尾继续追加 split 文件。阶段一的 A、B 清单由使用者显式提供；
代码不包含任何内置样本名单。准备好两个平铺目录后构建 35,632 个联合训练视图：

```bash
EXPECTED_COUNT=35632 bash scripts/build_joint_dataset.sh \
  /path/to/a_npy /path/to/b_stage1_npy /path/to/joint_ab_npy
```

默认创建相对软链接；设置 `MATERIALIZE_MODE=copy` 可以复制文件。阶段二使用完整
B 清单单独生成的 19,699 个 NPY，不需要加入联合目录。

## 两阶段训练

阶段一从外部初始权重加载模型参数，在 A+B 数据上训练 19 轮，固定选用
`epoch_0019_model.pkl`。阶段二只加载该模型参数、重新创建 Adam，在 Dataset B
上训练 9 轮，固定选用 `epoch_0009_model.pkl`。

```bash
bash scripts/train_two_stage.sh \
  /path/to/joint_ab_npy \
  /path/to/b_full_npy \
  /path/to/train_run \
  /path/to/initial_model.pkl \
  0
```

也可以分别使用 `train_stage1.sh` 和 `train_stage2.sh`。设置 `SMOKE=1` 时，每个
阶段只执行一次真实参数更新并保存、重新加载权重。2,048 点 patch 的 EMD 使用
确定性的前 1,024 点子集。

## 推理

随目录提供的最终权重为 `checkpoints/model_epoch_0009.pkl`。正式推理参数固定为：
patch 2048、patch batch 12、两轮去噪、`seed_k=8`、poly power 1、identity、
seed 2023、patch radius mode none。

单卡：

```bash
bash scripts/infer_single_gpu.sh /path/to/test_noisy /path/to/run 0
```

四卡：

```bash
GPU_IDS=0,1,2,3 bash scripts/infer_four_gpu.sh \
  /path/to/test_noisy /path/to/run
```

四卡入口默认使用 `nohup` 在后台执行并立即返回终端，启动成功时打印后台 PID、
运行目录和主日志路径。主日志为 `/path/to/run/logs/inference.log`；不额外启动
完成状态监控任务。

推理结束后会核对输入/输出 ID、数量、shape、dtype 和有限值。四卡入口使用固定
shard，合并时拒绝重复 ID；默认只生成 `predictions/`、`reports/` 和日志。

## 目录结构

```text
code/          Heavy 模型、EMD、数据、训练、推理和命令入口
checkpoints/   第 9 轮最终权重
configs/       阶段一、阶段二和推理固定参数
scripts/       环境、数据准备、训练和单卡/四卡推理入口
```
