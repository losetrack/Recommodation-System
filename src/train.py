import argparse
import importlib
import os
import pickle

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader

tqdm_module = importlib.util.find_spec("tqdm.auto")
if tqdm_module is not None:
	tqdm = importlib.import_module("tqdm.auto").tqdm
else:
	tqdm = None

from data_processer import (
	DENSE_FEATURES,
	SPARSE_FEATURES,
	CriteoPreprocessor,
	load_criteo_data,
)
from dataset import CriteoDataset
from model import DeepFM


def split_train_val(df, val_ratio=0.1):
	"""按时间顺序切分，避免未来信息泄漏。"""
	total = len(df)
	val_size = max(1, int(total * val_ratio))
	train_df = df.iloc[:-val_size].copy()
	val_df = df.iloc[-val_size:].copy()
	return train_df, val_df


def move_batch_to_device(x_dict, y, device):
	x_dict = {k: v.to(device) for k, v in x_dict.items()}
	y = y.to(device)
	return x_dict, y


def build_progress(iterable, enabled, desc):
	if enabled and tqdm is not None:
		return tqdm(iterable, desc=desc, leave=False)
	return iterable


def train_one_epoch(model, data_loader, criterion, optimizer, device, show_progress=True, epoch_idx=1):
	model.train()
	losses = []
	iterator = build_progress(data_loader, show_progress, f"Train Epoch {epoch_idx}")

	for x_dict, y in iterator:
		x_dict, y = move_batch_to_device(x_dict, y, device)

		optimizer.zero_grad()
		pred = model(x_dict).view(-1)
		loss = criterion(pred, y)
		loss.backward()
		optimizer.step()

		losses.append(loss.item())
		if tqdm is not None and hasattr(iterator, "set_postfix"):
			iterator.set_postfix(loss=f"{loss.item():.4f}")

	return float(np.mean(losses)) if losses else 0.0


@torch.no_grad()
def evaluate(model, data_loader, criterion, device, show_progress=True, epoch_idx=1):
	model.eval()
	losses = []
	y_true = []
	y_prob = []
	iterator = build_progress(data_loader, show_progress, f"Val Epoch {epoch_idx}")

	for x_dict, y in iterator:
		x_dict, y = move_batch_to_device(x_dict, y, device)

		pred = model(x_dict).view(-1)
		loss = criterion(pred, y)
		losses.append(loss.item())
		if tqdm is not None and hasattr(iterator, "set_postfix"):
			iterator.set_postfix(loss=f"{loss.item():.4f}")

		y_true.extend(y.detach().cpu().numpy().tolist())
		y_prob.extend(pred.detach().cpu().numpy().tolist())

	avg_loss = float(np.mean(losses)) if losses else 0.0

	# AUC 至少需要正负两类样本。
	auc = float("nan")
	if len(set(y_true)) > 1:
		auc = roc_auc_score(y_true, y_prob)

	return avg_loss, auc


@torch.no_grad()
def predict(model, data_loader, device, show_progress=True):
	model.eval()
	probs = []
	iterator = build_progress(data_loader, show_progress, "Predict")

	for x_dict in iterator:
		x_dict = {k: v.to(device) for k, v in x_dict.items()}
		pred = model(x_dict).view(-1)
		probs.extend(pred.detach().cpu().numpy().tolist())

	return probs


def main(args):
	os.makedirs(args.checkpoint_dir, exist_ok=True)

	feature_names = DENSE_FEATURES + SPARSE_FEATURES

	print("1) Loading training data...")
	raw_train_df = load_criteo_data(args.train_path, has_label=True)
	train_df, val_df = split_train_val(raw_train_df, val_ratio=args.val_ratio)

	print("2) Training set fitting, preprocessing and transformation of the validation set...")
	preprocessor = CriteoPreprocessor(num_bins=args.num_bins)
	train_processed, feature_vocab_sizes = preprocessor.fit_transform(train_df)
	val_processed = preprocessor.transform(val_df, has_label=True)

	train_dataset = CriteoDataset(train_processed, feature_names=feature_names, label_col="label")
	val_dataset = CriteoDataset(val_processed, feature_names=feature_names, label_col="label")

	train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
	val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)

	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
	model = DeepFM(
		feature_vocab_sizes=feature_vocab_sizes,
		embed_dim=args.embed_dim,
		dnn_hidden_units=args.hidden_units,
		dropout_rate=args.dropout,
	).to(device)

	criterion = nn.BCELoss()
	optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

	print("3) Start training...")
	best_auc = -1.0
	best_model_path = os.path.join(args.checkpoint_dir, "best_model.pt")

	for epoch in range(1, args.epochs + 1):
		train_loss = train_one_epoch(
			model,
			train_loader,
			criterion,
			optimizer,
			device,
			show_progress=not args.disable_progress,
			epoch_idx=epoch,
		)
		val_loss, val_auc = evaluate(
			model,
			val_loader,
			criterion,
			device,
			show_progress=not args.disable_progress,
			epoch_idx=epoch,
		)

		print(
			f"Epoch {epoch:02d} | train_loss={train_loss:.6f} | "
			f"val_loss={val_loss:.6f} | val_auc={val_auc:.6f}"
		)

		auc_for_select = val_auc if not np.isnan(val_auc) else -1.0
		if auc_for_select > best_auc:
			best_auc = auc_for_select
			torch.save(
				{
					"model_state_dict": model.state_dict(),
					"feature_vocab_sizes": feature_vocab_sizes,
					"feature_names": feature_names,
					"embed_dim": args.embed_dim,
					"hidden_units": args.hidden_units,
					"dropout": args.dropout,
				},
				best_model_path,
			)

	print(f"最优模型已保存到: {best_model_path}")

	preprocessor_path = os.path.join(args.checkpoint_dir, "preprocessor.pkl")
	with open(preprocessor_path, "wb") as f:
		pickle.dump(preprocessor, f)
	print(f"预处理器已保存到: {preprocessor_path}")

	if args.predict_test:
		print("4) 生成 test 预测结果...")
		test_df = load_criteo_data(args.test_path, has_label=False)
		test_processed = preprocessor.transform(test_df, has_label=False)
		test_dataset = CriteoDataset(test_processed, feature_names=feature_names, label_col="label")
		test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

		checkpoint = torch.load(best_model_path, map_location=device)
		model.load_state_dict(checkpoint["model_state_dict"])
		probs = predict(model, test_loader, device, show_progress=not args.disable_progress)

		output_path = os.path.join(args.checkpoint_dir, "test_predictions.txt")
		with open(output_path, "w", encoding="utf-8") as f:
			for p in probs:
				f.write(f"{p}\n")

		print(f"测试集预测已写入: {output_path}")


if __name__ == "__main__":
	parser = argparse.ArgumentParser(description="Train DeepFM on Criteo-like dataset")
	parser.add_argument("--train_path", type=str, default="./data/train.txt")
	parser.add_argument("--test_path", type=str, default="./data/test.txt")
	parser.add_argument("--checkpoint_dir", type=str, default="./checkpoints")

	parser.add_argument("--val_ratio", type=float, default=0.1)
	parser.add_argument("--num_bins", type=int, default=10)

	parser.add_argument("--epochs", type=int, default=3)
	parser.add_argument("--batch_size", type=int, default=2048)
	parser.add_argument("--lr", type=float, default=1e-3)
	parser.add_argument("--weight_decay", type=float, default=1e-6)

	parser.add_argument("--embed_dim", type=int, default=8)
	parser.add_argument("--hidden_units", type=int, nargs="+", default=[64, 32])
	parser.add_argument("--dropout", type=float, default=0.2)

	parser.add_argument("--predict_test", action="store_true")
	parser.add_argument("--disable_progress", action="store_true", help="Disable tqdm progress bars")

	args = parser.parse_args()
	main(args)
