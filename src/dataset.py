import torch
from torch.utils.data import Dataset, IterableDataset
from torch.utils.data import get_worker_info

from data_processer import DENSE_FEATURES, SPARSE_FEATURES


class CriteoDataset(Dataset):
	def __init__(self, df, feature_names, label_col="label"):
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
		return self.length

	def __getitem__(self, idx):
		x_dict = {feat: self.features[feat][idx] for feat in self.feature_names}
		if self.has_label:
			return x_dict, self.labels[idx]
		return x_dict


class CriteoStreamingDataset(IterableDataset):
	"""逐行读取样本并在线编码，避免一次性加载到内存。"""

	def __init__(self, file_path, preprocessor, has_label=True):
		self.file_path = file_path
		self.preprocessor = preprocessor
		self.has_label = has_label

		self.dense_features = DENSE_FEATURES
		self.sparse_features = SPARSE_FEATURES
		self.feature_names = self.dense_features + self.sparse_features

	def _safe_to_float(self, value):
		if value == "":
			return 0.0
		try:
			return float(value)
		except ValueError:
			return 0.0

	def _bucketize_dense_value(self, feat, value):
		edges = self.preprocessor.dense_bin_edges.get(feat)

		if edges is None:
			return 0

		idx = int(torch.searchsorted(torch.tensor(edges), torch.tensor(value), right=True).item()) - 1
		max_idx = len(edges) - 2
		if idx < 0:
			return 0
		if idx > max_idx:
			return max_idx
		return idx

	def _parse_line(self, line):
		parts = line.rstrip("\n").split("\t")
		expected = 1 + len(self.feature_names) if self.has_label else len(self.feature_names)

		if len(parts) < expected:
			parts.extend([""] * (expected - len(parts)))
		elif len(parts) > expected:
			parts = parts[:expected]

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
		label = 0.0 if label_raw == "" else float(label_raw)
		return x_dict, torch.tensor(label, dtype=torch.float32)

	def __iter__(self):
		worker_info = get_worker_info()
		worker_id = worker_info.id if worker_info is not None else 0
		num_workers = worker_info.num_workers if worker_info is not None else 1

		with open(self.file_path, "r", encoding="utf-8", errors="ignore") as f:
			for line_idx, line in enumerate(f):
				if line_idx % num_workers != worker_id:
					continue
				if not line.strip():
					continue
				yield self._parse_line(line)
