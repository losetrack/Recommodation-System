# Recommendation-System

一个基于 DeepFM 的点击率（CTR）预估项目，面向 Criteo 风格数据（13 个数值特征 + 26 个类别特征），支持从原始文本到训练、评估、预测的完整流程。

## 项目亮点

- 支持三种训练方式：内存模式、流式模式、NPZ 分片模式
- 预处理可复用：保存并加载 `preprocessor.pkl`
- 适配大数据：支持按块预处理与分片读取，降低内存压力
- 训练过程包含验证集评估与最优模型保存

## 目录结构

```text
Recommendation-System/
├─ data/
│  ├─ train.txt
│  ├─ test.txt
│  └─ readme.txt
├─ src/
│  ├─ data_loader.py
│  ├─ data_processer.py
│  ├─ dataset.py
│  ├─ model.py
│  ├─ train.py
│  └─ evaluate.py
├─ segment_sample.py
├─ split_train_val_stream.py
├─ preprocess_to_npz.py
└─ requirement.txt
```

## 环境安装

> 建议 Python 3.9+。

```bash
pip install -r requirement.txt
```

## 数据格式说明

参考 `data/readme.txt`（Criteo Display Advertising Challenge）。

- 训练集：`label + 13 个数值特征 + 26 个类别特征`
- 测试集：无 `label`
- 分隔符：Tab（`\t`）

## 快速开始

### 1) 可选：抽样小数据（快速验证流程）

```bash
python segment_sample.py --input ./data/train.txt --output ./data/train_small.txt --frac 0.01
```

### 2) 可选：先切分 train/val（用于流式训练）

```bash
python split_train_val_stream.py \
  --input ./data/train.txt \
  --train-output ./data/train_stream.txt \
  --val-output ./data/val_stream.txt \
  --val-ratio 0.1
```

### 3) 训练

#### 方式 A：内存模式（默认）

适合数据量较小，直接读 `train.txt` 并按时间顺序切分 train/val。

```bash
python src/train.py \
  --train_path ./data/train.txt \
  --checkpoint_dir ./checkpoints \
  --epochs 3 \
  --batch_size 2048
```

#### 方式 B：流式模式（推荐大文件文本）

先准备 `train_stream.txt` 与 `val_stream.txt`，再流式读取训练。

```bash
python src/train.py \
  --stream_train \
  --train_path ./data/train_stream.txt \
  --val_path ./data/val_stream.txt \
  --checkpoint_dir ./checkpoints \
  --epochs 3 \
  --batch_size 2048
```

#### 方式 C：NPZ 分片模式（推荐超大规模）

先将 train/val 文本预处理为 NPZ 分片，再训练。

1. 预处理 train 分片（会在输出目录生成 `preprocessor.pkl`）

```bash
python preprocess_to_npz.py \
  --input ./data/train_stream.txt \
  --output-dir ./data/npz_train \
  --has-label \
  --fit-path ./data/train_stream.txt \
  --chunk-size 200000
```

2. 预处理 val 分片（复用 train 的预处理器）

```bash
python preprocess_to_npz.py \
  --input ./data/val_stream.txt \
  --output-dir ./data/npz_val \
  --has-label \
  --preprocessor-path ./data/npz_train/preprocessor.pkl \
  --chunk-size 200000
```

3. 使用 NPZ 分片训练

```bash
python src/train.py \
  --npz_train \
  --train_npz_dir ./data/npz_train \
  --val_npz_dir ./data/npz_val \
  --checkpoint_dir ./checkpoints \
  --epochs 3 \
  --batch_size 2048
```

## 评估与预测

### 文本数据评估/预测

```bash
python src/evaluate.py \
  --checkpoint_dir ./checkpoints \
  --data_path ./data/test.txt \
  --output_path ./checkpoints/predictions.txt
```

如果输入数据包含标签，可加 `--has_label` 输出 AUC：

```bash
python src/evaluate.py \
  --checkpoint_dir ./checkpoints \
  --data_path ./data/val_stream.txt \
  --output_path ./checkpoints/val_predictions.txt \
  --has_label
```

### NPZ 数据评估/预测

```bash
python src/evaluate.py \
  --checkpoint_dir ./checkpoints \
  --npz_input_dir ./data/npz_val \
  --output_path ./checkpoints/npz_val_predictions.txt \
  --has_label
```

## 输出文件

训练后默认在 `checkpoints/` 生成：

- `best_model.pt`：最优模型参数
- `preprocessor.pkl`：预处理器状态（分箱、哈希配置等）
- `test_predictions.txt`：若训练时启用 `--predict_test` 则生成

## 常用参数

- `--num_bins`：数值特征分箱数（默认 10）
- `--hash_dim`：类别特征哈希空间大小（默认 2^20）
- `--embed_dim`：Embedding 维度（默认 8）
- `--hidden_units`：DNN 隐藏层（默认 64 32）
- `--dropout`：Dropout 比例（默认 0.2）

## 注意事项

- 项目依赖文件名为 `requirements.txt`。
- 推荐在仓库根目录执行命令。
- 使用流式模式时，建议显式提供独立的 `--val_path`，避免数据泄漏。

## 致谢

数据集说明来源于 Criteo Display Advertising Challenge。
