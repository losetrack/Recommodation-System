import os

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from data_processer import (
	DENSE_FEATURES,
	SPARSE_FEATURES,
	CriteoPreprocessor,
	load_criteo_data,
)
from dataset import CriteoStreamingDataset


def split_train_val(df, val_ratio=0.1):
	"""按时间顺序切分，避免未来信息泄漏。"""
	total = len(df)
	val_size = max(1, int(total * val_ratio))
	train_df = df.iloc[:-val_size].copy()
	val_df = df.iloc[-val_size:].copy()
	return train_df, val_df


class CriteoArrayDataset(Dataset):
	"""基于 numpy 数组构建 Dataset。"""

	def __init__(self, dense, sparse, labels=None):
		if dense.shape[0] != sparse.shape[0]:
			raise ValueError("dense 和 sparse 样本数不一致")

		if labels is not None and len(labels) != dense.shape[0]:
			raise ValueError("labels 与特征样本数不一致")

		self.has_label = labels is not None

		# dense 是分箱后的索引，模型侧按离散特征处理。
		self.dense = torch.as_tensor(dense.astype(np.int64, copy=True), dtype=torch.long)
		self.sparse = torch.as_tensor(sparse.astype(np.int64, copy=True), dtype=torch.long)

		self.labels = None
		if self.has_label:
			self.labels = torch.as_tensor(labels.astype(np.float32, copy=True), dtype=torch.float32)

		self.length = self.dense.shape[0]

	def __len__(self):
		return self.length

	def __getitem__(self, idx):
		x_dict = {}

		for i, feat in enumerate(DENSE_FEATURES):
			x_dict[feat] = self.dense[idx, i]

		for i, feat in enumerate(SPARSE_FEATURES):
			x_dict[feat] = self.sparse[idx, i]

		if self.has_label:
			return x_dict, self.labels[idx]
		return x_dict


def build_feature_vocab_sizes(preprocessor):
	"""根据预处理器状态构建模型 embedding vocab 大小。"""
	feature_vocab_sizes = {}

	for feat in DENSE_FEATURES:
		edges = preprocessor.dense_bin_edges.get(feat)
		if edges is None:
			feature_vocab_sizes[feat] = 1
		else:
			feature_vocab_sizes[feat] = len(edges) - 1

	for feat in SPARSE_FEATURES:
		# 稀疏特征经过哈希后落在 [0, hash_dim) 区间。
		feature_vocab_sizes[feat] = preprocessor.hash_dim

	return feature_vocab_sizes


def _build_array_dataloader(dense, sparse, labels, batch_size, shuffle, num_workers):
	dataset = CriteoArrayDataset(dense=dense, sparse=sparse, labels=labels)
	return DataLoader(
		dataset,
		batch_size=batch_size,
		shuffle=shuffle,
		num_workers=num_workers,
		pin_memory=torch.cuda.is_available(),
	)


def build_in_memory_train_val_loaders(
	train_path,
	val_ratio=0.1,
	num_bins=10,
	hash_dim=2**20,
	batch_size=2048,
	num_workers=0,
):
	"""构建内存版 train/val DataLoader。"""
	raw_train_df = load_criteo_data(train_path, has_label=True)
	train_df, val_df = split_train_val(raw_train_df, val_ratio=val_ratio)

	preprocessor = CriteoPreprocessor(num_bins=num_bins, hash_dim=hash_dim)
	preprocessor.fit(train_df)

	train_dense, train_sparse, train_label = preprocessor.transform(train_df, has_label=True)
	val_dense, val_sparse, val_label = preprocessor.transform(val_df, has_label=True)

	train_loader = _build_array_dataloader(
		dense=train_dense,
		sparse=train_sparse,
		labels=train_label,
		batch_size=batch_size,
		shuffle=True,
		num_workers=num_workers,
	)

	val_loader = _build_array_dataloader(
		dense=val_dense,
		sparse=val_sparse,
		labels=val_label,
		batch_size=batch_size,
		shuffle=False,
		num_workers=num_workers,
	)

	feature_vocab_sizes = build_feature_vocab_sizes(preprocessor)
	return train_loader, val_loader, preprocessor, feature_vocab_sizes


def build_streaming_loader(file_path, preprocessor, has_label=True, batch_size=2048, num_workers=0):
	"""构建流式 DataLoader，逐行读取文件。"""
	if not os.path.exists(file_path):
		raise FileNotFoundError(f"文件不存在: {file_path}")

	dataset = CriteoStreamingDataset(
		file_path=file_path,
		preprocessor=preprocessor,
		has_label=has_label,
	)

	return DataLoader(
		dataset,
		batch_size=batch_size,
		shuffle=False,
		num_workers=num_workers,
		pin_memory=torch.cuda.is_available(),
	)
