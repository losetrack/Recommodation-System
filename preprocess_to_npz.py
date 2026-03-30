import argparse
import json
import os
import pickle
import sys

import numpy as np
import pandas as pd

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
SRC_DIR = os.path.join(ROOT_DIR, "src")
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

from data_processer import CriteoPreprocessor, get_criteo_columns


def build_parser():
    """构建命令行参数解析器。"""
    parser = argparse.ArgumentParser(description="Preprocess Criteo-like TSV data into chunked NPZ shards.")
    parser.add_argument("--input", type=str, required=True, help="Path to source TSV file")
    parser.add_argument("--output-dir", type=str, required=True, help="Directory to write NPZ shards")
    parser.add_argument("--has-label", action="store_true", help="Whether the input contains the label column")
    parser.add_argument("--chunk-size", type=int, default=200000, help="Rows per output shard")
    parser.add_argument("--compressed", action="store_true", help="Use np.savez_compressed instead of np.savez")

    parser.add_argument("--preprocessor-path", type=str, default="", help="Existing preprocessor.pkl to reuse")
    parser.add_argument("--fit-path", type=str, default="", help="File used to fit a new preprocessor")
    parser.add_argument("--fit-rows", type=int, default=1000000, help="Rows to read from fit-path when creating a new preprocessor")
    parser.add_argument("--num-bins", type=int, default=10, help="Dense feature bin count when fitting a new preprocessor")
    parser.add_argument("--hash-dim", type=int, default=2**20, help="Hash space size when fitting a new preprocessor")
    return parser


def validate_args(args):
    """校验预处理脚本参数的有效性。"""
    if not os.path.exists(args.input):
        raise FileNotFoundError(f"Input file not found: {args.input}")
    if args.chunk_size <= 0:
        raise ValueError(f"--chunk-size must be > 0, got {args.chunk_size}")
    if args.fit_rows <= 0:
        raise ValueError(f"--fit-rows must be > 0, got {args.fit_rows}")
    if not args.preprocessor_path and not args.fit_path:
        raise ValueError("Provide either --preprocessor-path or --fit-path.")
    if args.preprocessor_path and not os.path.exists(args.preprocessor_path):
        raise FileNotFoundError(f"Preprocessor file not found: {args.preprocessor_path}")
    if args.fit_path and not os.path.exists(args.fit_path):
        raise FileNotFoundError(f"Fit file not found: {args.fit_path}")


def load_preprocessor(args):
    """加载已有预处理器，或基于样本数据拟合新预处理器。"""
    if args.preprocessor_path:
        with open(args.preprocessor_path, "rb") as f:
            return pickle.load(f)

    fit_path = args.fit_path or args.input
    print(f"Fitting preprocessor from: {fit_path}")
    fit_df = pd.read_csv(
        fit_path,
        sep="\t",
        header=None,
        names=get_criteo_columns(has_label=True),
        nrows=args.fit_rows,
    )
    preprocessor = CriteoPreprocessor(num_bins=args.num_bins, hash_dim=args.hash_dim)
    preprocessor.fit(fit_df)
    return preprocessor


def iter_chunks(file_path, has_label, chunk_size):
    """按块读取 TSV 文件并逐块产出 DataFrame。"""
    yield from pd.read_csv(
        file_path,
        sep="\t",
        header=None,
        names=get_criteo_columns(has_label=has_label),
        chunksize=chunk_size,
    )


def save_shard(output_dir, shard_idx, dense, sparse, labels, compressed):
    """将单个特征分片保存为 NPZ 文件并返回元信息。"""
    save_fn = np.savez_compressed if compressed else np.savez
    shard_name = f"part-{shard_idx:05d}.npz"
    shard_path = os.path.join(output_dir, shard_name)

    payload = {
        "dense": dense.astype(np.int64, copy=False),
        "sparse": sparse.astype(np.int64, copy=False),
    }
    if labels is not None:
        payload["labels"] = labels.astype(np.float32, copy=False)

    save_fn(shard_path, **payload)
    return shard_name, int(dense.shape[0])


def write_manifest(output_dir, manifest):
    """将分片清单写入 manifest.json。"""
    manifest_path = os.path.join(output_dir, "manifest.json")
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
    return manifest_path


def main():
    """执行数据预处理主流程并产出 NPZ 分片与清单。"""
    parser = build_parser()
    args = parser.parse_args()
    validate_args(args)

    os.makedirs(args.output_dir, exist_ok=True)
    preprocessor = load_preprocessor(args)

    preprocessor_path = os.path.join(args.output_dir, "preprocessor.pkl")
    with open(preprocessor_path, "wb") as f:
        pickle.dump(preprocessor, f)

    manifest = {
        "input": args.input,
        "has_label": args.has_label,
        "chunk_size": args.chunk_size,
        "compressed": args.compressed,
        "preprocessor_path": preprocessor_path,
        "num_shards": 0,
        "num_rows": 0,
        "shards": [],
    }

    for shard_idx, chunk_df in enumerate(iter_chunks(args.input, args.has_label, args.chunk_size)):
        transformed = preprocessor.transform(chunk_df, has_label=args.has_label)
        if args.has_label:
            dense, sparse, labels = transformed
        else:
            dense, sparse = transformed
            labels = None

        shard_name, rows = save_shard(
            output_dir=args.output_dir,
            shard_idx=shard_idx,
            dense=dense,
            sparse=sparse,
            labels=labels,
            compressed=args.compressed,
        )
        manifest["num_shards"] += 1
        manifest["num_rows"] += rows
        manifest["shards"].append({"file": shard_name, "rows": rows})
        print(f"Wrote {shard_name} with {rows} rows")

    manifest_path = write_manifest(args.output_dir, manifest)
    print(f"Saved preprocessor to: {preprocessor_path}")
    print(f"Saved manifest to: {manifest_path}")
    print(f"Done. shards={manifest['num_shards']}, rows={manifest['num_rows']}")


if __name__ == "__main__":
    main()
