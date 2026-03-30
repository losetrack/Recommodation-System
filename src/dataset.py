import os
import random
import json

import numpy as np
import torch
from torch.utils.data import Dataset, IterableDataset
from torch.utils.data import get_worker_info

from data_processer import DENSE_FEATURES, SPARSE_FEATURES


class CriteoDataset(Dataset):
	def __init__(self, df, feature_names, label_col="label"):
		"""用内存 DataFrame 初始化基础监督数据集。"""
		self.feature_names = feature_names
		self.label_col = label_col
		self.has_label = label_col in df.columns
		self.df = df.copy()

		self.features = {
			feat: torch.as_tensor(df[feat].to_numpy(copy=True), dtype=torch.long)
			for feat in self.feature_names
		}

		self.labels = None
		if self.has_label:
			self.labels = torch.as_tensor(df[self.label_col].to_numpy(copy=True), dtype=torch.float32)

		self.length = len(df)

	def __len__(self):
		"""返回样本总数。"""
		return self.length

	def __getitem__(self, idx):
		"""按索引返回特征字典与可选标签。"""
		x_dict = {feat: self.features[feat][idx] for feat in self.feature_names}
		if self.has_label:
			return x_dict, self.labels[idx]
		return x_dict


class CriteoStreamingDataset(IterableDataset):
	"""逐行读取样本并在线编码，避免一次性加载到内存。"""

	def __init__(
		self,
		file_path,
		preprocessor,
		has_label=True,
		strict_schema=True,
		shuffle_buffer_size=0,
		seed=42,
		num_samples=None,
	):
		"""初始化流式数据集配置与预处理缓存。"""
		self.file_path = file_path
		self.preprocessor = preprocessor
		self.has_label = has_label
		self.strict_schema = strict_schema
		self.shuffle_buffer_size = max(0, int(shuffle_buffer_size))
		self.seed = int(seed)
		self.num_samples = num_samples

		self.dense_features = DENSE_FEATURES
		self.sparse_features = SPARSE_FEATURES
		self.feature_names = self.dense_features + self.sparse_features
		self.expected_fields = 1 + len(self.feature_names) if self.has_label else len(self.feature_names)
		self._dense_edges = {
			feat: None if preprocessor.dense_bin_edges.get(feat) is None else np.asarray(preprocessor.dense_bin_edges[feat])
			for feat in self.dense_features
		}

	def _safe_to_float(self, value):
		"""将字符串安全转换为浮点数，异常值回退为 0。"""
		if value == "":
			return 0.0
		try:
			return float(value)
		except ValueError:
			return 0.0

	def _bucketize_dense_value(self, feat, value):
		"""按已拟合分箱边界将连续值映射为桶索引。"""
		edges = self._dense_edges.get(feat)

		if edges is None:
			return 0

		idx = int(np.searchsorted(edges, value, side="right")) - 1
		max_idx = len(edges) - 2
		if idx < 0:
			return 0
		if idx > max_idx:
			return max_idx
		return idx

	def _parse_label(self, value, line_number):
		"""解析并校验标签字段。"""
		if value == "":
			if self.strict_schema:
				raise ValueError(f"line {line_number}: empty label")
			return 0.0
		try:
			return float(value)
		except ValueError as exc:
			raise ValueError(f"line {line_number}: invalid label {value!r}") from exc

	def _parse_line(self, line, line_number):
		"""将单行文本解析为模型输入特征与可选标签。"""
		parts = line.rstrip("\n").split("\t")

		if len(parts) != self.expected_fields:
			message = (
				f"line {line_number}: expected {self.expected_fields} tab-separated fields, "
				f"got {len(parts)}"
			)
			if self.strict_schema:
				raise ValueError(message)
			if len(parts) < self.expected_fields:
				parts.extend([""] * (self.expected_fields - len(parts)))
			else:
				parts = parts[:self.expected_fields]

		offset = 1 if self.has_label else 0
		x_dict = {}

		for i, feat in enumerate(self.dense_features):
			raw_value = parts[offset + i]
			value = self._safe_to_float(raw_value)
			bucket = self._bucketize_dense_value(feat, value)
			x_dict[feat] = torch.tensor(bucket, dtype=torch.long)

		base = offset + len(self.dense_features)
		for i, feat in enumerate(self.sparse_features):
			raw_value = parts[base + i] if base + i < len(parts) else ""
			value = raw_value if raw_value != "" else "-1"
			hashed = self.preprocessor._hash(feat, value)
			x_dict[feat] = torch.tensor(hashed, dtype=torch.long)

		if not self.has_label:
			return x_dict

		label_raw = parts[0] if parts else "0"
		label = self._parse_label(label_raw, line_number)
		return x_dict, torch.tensor(label, dtype=torch.float32)

	def __len__(self):
		"""返回样本数；未知长度时抛出异常。"""
		if self.num_samples is None:
			raise TypeError("Streaming dataset length is unknown; pass num_samples when constructing the loader.")
		return self.num_samples

	def _iter_file_chunk(self):
		"""按 worker 分片范围迭代二进制文件行。"""
		worker_info = get_worker_info()
		worker_id = worker_info.id if worker_info is not None else 0
		num_workers = worker_info.num_workers if worker_info is not None else 1

		file_size = os.path.getsize(self.file_path)
		chunk_size = (file_size + num_workers - 1) // num_workers
		start = worker_id * chunk_size
		end = min(file_size, start + chunk_size)

		with open(self.file_path, "rb") as f:
			if start > 0:
				f.seek(start - 1)
				f.readline()
			position = f.tell()

			while position < end:
				raw_line = f.readline()
				if not raw_line:
					break
				position = f.tell()
				line_number = None
				line = raw_line.decode("utf-8", errors="ignore")
				if not line.strip():
					continue
				yield line_number, line

	def _iter_decoded_lines(self):
		"""将文件分片内容解码为带行号的文本流。"""
		for fallback_idx, (line_number, line) in enumerate(self._iter_file_chunk(), start=1):
			yield fallback_idx if line_number is None else line_number, line

	def _shuffle_stream(self, iterable, rng):
		"""使用缓冲区对流式样本做近似随机打乱。"""
		if self.shuffle_buffer_size <= 1:
			yield from iterable
			return

		buffer = []

		for item in iterable:
			buffer.append(item)
			if len(buffer) >= self.shuffle_buffer_size:
				idx = rng.randrange(len(buffer))
				yield buffer.pop(idx)

		while buffer:
			idx = rng.randrange(len(buffer))
			yield buffer.pop(idx)

	def __iter__(self):
		"""迭代输出流式编码后的样本。"""
		worker_info = get_worker_info()
		worker_id = worker_info.id if worker_info is not None else 0
		rng = random.Random(self.seed + worker_id + int(torch.initial_seed()))

		line_iter = self._iter_decoded_lines()
		if self.shuffle_buffer_size > 1:
			line_iter = self._shuffle_stream(line_iter, rng)

		for line_number, line in line_iter:
			yield self._parse_line(line, line_number)


class CriteoNPZDataset(IterableDataset):
	"""按 shard 读取离线编码后的 NPZ 数据。"""

	def __init__(self, manifest_path, shuffle_shards=False, shuffle_samples=False, seed=42):
		"""读取 manifest 并初始化 NPZ 分片迭代配置。"""
		if not os.path.exists(manifest_path):
			raise FileNotFoundError(f"NPZ manifest not found: {manifest_path}")

		self.manifest_path = manifest_path
		with open(manifest_path, "r", encoding="utf-8") as f:
			self.manifest = json.load(f)

		self.root_dir = os.path.dirname(os.path.abspath(manifest_path))
		self.has_label = bool(self.manifest.get("has_label", False))
		self.shuffle_shards = shuffle_shards
		self.shuffle_samples = shuffle_samples
		self.seed = int(seed)
		self.shards = list(self.manifest.get("shards", []))

		if not self.shards:
			raise ValueError(f"No shards found in manifest: {manifest_path}")

	def __len__(self):
		"""返回 manifest 中记录的样本总数。"""
		return int(self.manifest.get("num_rows", 0))

	def _ordered_shards(self):
		"""按 worker 规则返回当前进程负责的分片顺序。"""
		shards = list(self.shards)
		worker_info = get_worker_info()
		worker_id = worker_info.id if worker_info is not None else 0
		num_workers = worker_info.num_workers if worker_info is not None else 1

		if self.shuffle_shards:
			rng = random.Random(self.seed + worker_id + int(torch.initial_seed()))
			rng.shuffle(shards)

		return shards[worker_id::num_workers]

	def _load_shard(self, shard_file):
		"""加载单个 NPZ 分片文件。"""
		shard_path = os.path.join(self.root_dir, shard_file)
		if not os.path.exists(shard_path):
			raise FileNotFoundError(f"NPZ shard not found: {shard_path}")
		return np.load(shard_path)

	def _build_item(self, dense, sparse, idx, labels=None):
		"""从分片数组中构造单条模型输入样本。"""
		x_dict = {}
		for i, feat in enumerate(DENSE_FEATURES):
			x_dict[feat] = torch.tensor(int(dense[idx, i]), dtype=torch.long)
		for i, feat in enumerate(SPARSE_FEATURES):
			x_dict[feat] = torch.tensor(int(sparse[idx, i]), dtype=torch.long)

		if labels is None:
			return x_dict
		return x_dict, torch.tensor(float(labels[idx]), dtype=torch.float32)

	def __iter__(self):
		"""按分片与样本顺序迭代输出训练样本。"""
		worker_info = get_worker_info()
		worker_id = worker_info.id if worker_info is not None else 0
		rng = random.Random(self.seed + worker_id + int(torch.initial_seed()))

		for shard in self._ordered_shards():
			with self._load_shard(shard["file"]) as data:
				dense = data["dense"]
				sparse = data["sparse"]
				labels = data["labels"] if self.has_label and "labels" in data else None

				num_rows = dense.shape[0]
				indices = list(range(num_rows))
				if self.shuffle_samples:
					rng.shuffle(indices)

				for idx in indices:
					yield self._build_item(dense, sparse, idx, labels=labels)
