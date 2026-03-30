import argparse
import importlib
import os
import pickle

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import roc_auc_score

tqdm_module = importlib.util.find_spec("tqdm.auto")
if tqdm_module is not None:
	tqdm = importlib.import_module("tqdm.auto").tqdm
else:
	tqdm = None

from data_loader import (
	build_feature_vocab_sizes,
	build_in_memory_train_val_loaders,
	build_npz_loader,
	build_streaming_loader,
	load_npz_bundle,
)
from data_processer import CriteoPreprocessor, load_criteo_data
from model import DeepFM

def move_batch_to_device(x_dict, y, device):
	"""将一个 batch 的特征与标签移动到目标设备。"""
	x_dict = {k: v.to(device) for k, v in x_dict.items()}
	y = y.to(device)
	return x_dict, y


def build_progress(iterable, enabled, desc):
	"""按需包装 tqdm 进度条，不可用时回退原迭代器。"""
	if enabled and tqdm is not None:
		return tqdm(iterable, desc=desc, leave=False)
	return iterable


def train_one_epoch(model, data_loader, criterion, optimizer, device, show_progress=True, epoch_idx=1):
	"""执行一轮训练并返回平均损失。"""
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
	"""在验证集上评估模型并返回损失与 AUC。"""
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
	"""对无标签数据执行推理并返回预测概率列表。"""
	model.eval()
	probs = []
	iterator = build_progress(data_loader, show_progress, "Predict")

	for x_dict in iterator:
		x_dict = {k: v.to(device) for k, v in x_dict.items()}
		pred = model(x_dict).view(-1)
		probs.extend(pred.detach().cpu().numpy().tolist())

	return probs


def load_or_fit_preprocessor(args):
	"""加载已有预处理器，或根据训练集重新拟合预处理器。"""
	if args.preprocessor_path:
		if not os.path.exists(args.preprocessor_path):
			raise FileNotFoundError(f"未找到预处理器文件: {args.preprocessor_path}")
		with open(args.preprocessor_path, "rb") as f:
			return pickle.load(f)

	print("2) Fitting preprocessor from train data...")
	raw_train_df = load_criteo_data(args.train_path, has_label=True)
	preprocessor = CriteoPreprocessor(num_bins=args.num_bins, hash_dim=args.hash_dim)
	preprocessor.fit(raw_train_df)
	return preprocessor


def main(args):
	"""训练 DeepFM 主流程，支持内存、流式与 NPZ 三种数据模式。"""
	os.makedirs(args.checkpoint_dir, exist_ok=True)

	print("1) Loading data via data_loader...")
	if args.npz_train:
		if not args.train_npz_dir or not args.val_npz_dir:
			raise ValueError("NPZ 训练模式需要显式提供 --train_npz_dir 和 --val_npz_dir。")

		_, preprocessor = load_npz_bundle(args.train_npz_dir)
		feature_vocab_sizes = build_feature_vocab_sizes(preprocessor)
		train_loader, train_manifest = build_npz_loader(
			npz_dir=args.train_npz_dir,
			batch_size=args.batch_size,
			num_workers=args.num_workers,
			shuffle=True,
			seed=args.seed,
		)
		val_loader, _ = build_npz_loader(
			npz_dir=args.val_npz_dir,
			batch_size=args.batch_size,
			num_workers=args.num_workers,
			shuffle=False,
			seed=args.seed,
		)
		print(
			f"2) Loaded NPZ shards: train_rows={train_manifest.get('num_rows', 'unknown')} "
			f"train_shards={train_manifest.get('num_shards', 'unknown')}"
		)
	elif args.stream_train:
		if not args.val_path:
			raise ValueError("流式训练模式需要显式提供 --val_path，避免从同一文件中泄漏验证集。")

		preprocessor = load_or_fit_preprocessor(args)
		feature_vocab_sizes = build_feature_vocab_sizes(preprocessor)
		train_loader = build_streaming_loader(
			file_path=args.train_path,
			preprocessor=preprocessor,
			has_label=True,
			batch_size=args.batch_size,
			num_workers=args.num_workers,
			strict_schema=not args.allow_bad_lines,
			shuffle_buffer_size=args.stream_shuffle_buffer_size,
			seed=args.seed,
			num_samples=args.train_num_samples,
		)
		val_loader = build_streaming_loader(
			file_path=args.val_path,
			preprocessor=preprocessor,
			has_label=True,
			batch_size=args.batch_size,
			num_workers=args.num_workers,
			strict_schema=not args.allow_bad_lines,
			shuffle_buffer_size=0,
			seed=args.seed,
			num_samples=args.val_num_samples,
		)
	else:
		train_loader, val_loader, preprocessor, feature_vocab_sizes = build_in_memory_train_val_loaders(
			train_path=args.train_path,
			val_ratio=args.val_ratio,
			num_bins=args.num_bins,
			hash_dim=args.hash_dim,
			batch_size=args.batch_size,
			num_workers=args.num_workers,
		)

	feature_names = list(feature_vocab_sizes.keys())

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
		print("4) 生成 test 预测结果（streaming）...")
		test_loader = build_streaming_loader(
			file_path=args.test_path,
			preprocessor=preprocessor,
			has_label=False,
			batch_size=args.batch_size,
			num_workers=args.num_workers,
		)

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
	parser.add_argument("--val_path", type=str, default="", help="Validation file path for streaming mode")
	parser.add_argument("--checkpoint_dir", type=str, default="./checkpoints")
	parser.add_argument("--preprocessor_path", type=str, default="", help="Reuse a fitted preprocessor for streaming mode")
	parser.add_argument("--train_npz_dir", type=str, default="", help="Directory containing train NPZ shards and manifest")
	parser.add_argument("--val_npz_dir", type=str, default="", help="Directory containing val NPZ shards and manifest")

	parser.add_argument("--val_ratio", type=float, default=0.1)
	parser.add_argument("--num_bins", type=int, default=10)
	parser.add_argument("--hash_dim", type=int, default=2**20)
	parser.add_argument("--num_workers", type=int, default=0)
	parser.add_argument("--seed", type=int, default=42)

	parser.add_argument("--epochs", type=int, default=3)
	parser.add_argument("--batch_size", type=int, default=2048)
	parser.add_argument("--lr", type=float, default=1e-3)
	parser.add_argument("--weight_decay", type=float, default=1e-6)

	parser.add_argument("--embed_dim", type=int, default=8)
	parser.add_argument("--hidden_units", type=int, nargs="+", default=[64, 32])
	parser.add_argument("--dropout", type=float, default=0.2)

	parser.add_argument("--stream_train", action="store_true", help="Use streaming dataloaders for train/val")
	parser.add_argument("--npz_train", action="store_true", help="Use offline NPZ shard dataloaders for train/val")
	parser.add_argument("--stream_shuffle_buffer_size", type=int, default=0, help="Shuffle buffer size for streaming train loader")
	parser.add_argument("--train_num_samples", type=int, default=None, help="Optional train sample count for streaming dataset length")
	parser.add_argument("--val_num_samples", type=int, default=None, help="Optional validation sample count for streaming dataset length")
	parser.add_argument("--allow_bad_lines", action="store_true", help="Pad/truncate malformed streaming rows instead of raising")
	parser.add_argument("--predict_test", action="store_true")
	parser.add_argument("--disable_progress", action="store_true", help="Disable tqdm progress bars")

	args = parser.parse_args()
	main(args)
