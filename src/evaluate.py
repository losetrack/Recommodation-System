import argparse
import importlib
import os
import pickle

import torch
from sklearn.metrics import roc_auc_score

from data_loader import build_streaming_loader
from model import DeepFM

tqdm_module = importlib.util.find_spec("tqdm.auto")
if tqdm_module is not None:
	tqdm = importlib.import_module("tqdm.auto").tqdm
else:
	tqdm = None


def build_progress(iterable, enabled, desc):
	if enabled and tqdm is not None:
		return tqdm(iterable, desc=desc, leave=False)
	return iterable


@torch.no_grad()
def predict(model, data_loader, device, show_progress=True):
	model.eval()
	probs = []
	y_true = []
	iterator = build_progress(data_loader, show_progress, "Evaluate")

	for batch in iterator:
		if isinstance(batch, (tuple, list)) and len(batch) == 2:
			x_dict = batch[0]
			y = batch[1]
			y_true.extend(y.detach().cpu().numpy().tolist())
		else:
			x_dict = batch

		x_dict = {k: v.to(device) for k, v in x_dict.items()}
		pred = model(x_dict).view(-1)
		probs.extend(pred.detach().cpu().numpy().tolist())

	return probs, y_true


def main(args):
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

	loader = build_streaming_loader(
		file_path=args.data_path,
		preprocessor=preprocessor,
		has_label=args.has_label,
		batch_size=args.batch_size,
		num_workers=args.num_workers,
	)

	probs, y_true = predict(model, loader, device, show_progress=not args.disable_progress)

	with open(args.output_path, "w", encoding="utf-8") as f:
		for p in probs:
			f.write(f"{p}\n")

	print(f"预测结果已输出到: {args.output_path}")

	if args.has_label:
		if len(set(y_true)) > 1:
			auc = roc_auc_score(y_true, probs)
			print(f"AUC: {auc:.6f}")
		else:
			print("AUC 无法计算：标签只有单一类别。")


if __name__ == "__main__":
	parser = argparse.ArgumentParser(description="Evaluate or predict with trained DeepFM")
	parser.add_argument("--checkpoint_dir", type=str, default="./checkpoints")
	parser.add_argument("--data_path", type=str, default="./data/test.txt")
	parser.add_argument("--output_path", type=str, default="./checkpoints/predictions.txt")
	parser.add_argument("--batch_size", type=int, default=2048)
	parser.add_argument("--num_workers", type=int, default=0)
	parser.add_argument("--disable_progress", action="store_true", help="Disable tqdm progress bars")
	parser.add_argument("--has_label", action="store_true")
	args = parser.parse_args()

	main(args)
