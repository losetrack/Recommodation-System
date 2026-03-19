import torch
from torch.utils.data import Dataset


class CriteoDataset(Dataset):
	def __init__(self, df, feature_names, label_col="label"):
		self.feature_names = feature_names
		self.label_col = label_col
		self.has_label = label_col in df.columns

		self.features = {
			feat: torch.as_tensor(df[feat].values, dtype=torch.long)
			for feat in self.feature_names
		}

		self.labels = None
		if self.has_label:
			self.labels = torch.as_tensor(df[self.label_col].values, dtype=torch.float32)

		self.length = len(df)

	def __len__(self):
		return self.length

	def __getitem__(self, idx):
		x_dict = {feat: self.features[feat][idx] for feat in self.feature_names}
		if self.has_label:
			return x_dict, self.labels[idx]
		return x_dict
