import argparse
import os
import pickle

import numpy as np
import torch
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader

from data_processer import DENSE_FEATURES, SPARSE_FEATURES, load_criteo_data
from dataset import CriteoDataset
from model import DeepFM


@torch.no_grad()
def predict(model, data_loader, device):
	model.eval()
	probs = []

	for batch in data_loader:
		if isinstance(batch, tuple):
			x_dict = batch[0]
		else:
			x_dict = batch

		x_dict = {k: v.to(device) for k, v in x_dict.items()}
		pred = model(x_dict).view(-1)
		probs.extend(pred.detach().cpu().numpy().tolist())

	return probs


def main(args):
	feature_names = DENSE_FEATURES + SPARSE_FEATURES
	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

	model_path = os.path.join(args.checkpoint_dir, "best_model.pt")
	preprocessor_path = os.path.join(args.checkpoint_dir, "preprocessor.pkl")

	if not os.path.exists(model_path):
		raise FileNotFoundError(f"未找到模型文件: {model_path}")
	if not os.path.exists(preprocessor_path):
		raise FileNotFoundError(f"未找到预处理器文件: {preprocessor_path}")

	with open(preprocessor_path, "rb") as f:
		preprocessor = pickle.load(f)

	checkpoint = torch.load(model_path, map_location=device)

	model = DeepFM(
		feature_vocab_sizes=checkpoint["feature_vocab_sizes"],
		embed_dim=checkpoint["embed_dim"],
		dnn_hidden_units=checkpoint["hidden_units"],
		dropout_rate=checkpoint["dropout"],
	).to(device)
	model.load_state_dict(checkpoint["model_state_dict"])

	raw_df = load_criteo_data(args.data_path, has_label=args.has_label)
	processed_df = preprocessor.transform(raw_df, has_label=args.has_label)

	dataset = CriteoDataset(processed_df, feature_names=feature_names, label_col="label")
	loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False)

	probs = predict(model, loader, device)

	with open(args.output_path, "w", encoding="utf-8") as f:
		for p in probs:
			f.write(f"{p}\n")

	print(f"预测结果已输出到: {args.output_path}")

	if args.has_label:
		labels = processed_df["label"].astype(float).tolist()
		if len(set(labels)) > 1:
			auc = roc_auc_score(labels, probs)
			print(f"AUC: {auc:.6f}")
		else:
			print("AUC 无法计算：标签只有单一类别。")


if __name__ == "__main__":
	parser = argparse.ArgumentParser(description="Evaluate or predict with trained DeepFM")
	parser.add_argument("--checkpoint_dir", type=str, default="./checkpoints")
	parser.add_argument("--data_path", type=str, default="./data/test.txt")
	parser.add_argument("--output_path", type=str, default="./checkpoints/predictions.txt")
	parser.add_argument("--batch_size", type=int, default=2048)
	parser.add_argument("--has_label", action="store_true")
	args = parser.parse_args()

	main(args)
