import argparse
import os


def build_parser():
    """构建抽样脚本命令行参数。"""
    parser = argparse.ArgumentParser(description="Stream-copy a small subset from Criteo train.txt")
    parser.add_argument("--input", type=str, default="./data/train.txt", help="Path to original train.txt")
    parser.add_argument("--output", type=str, default="./data/train_small.txt", help="Path to output subset file")
    parser.add_argument("--frac", type=float, default=0.01, help="Sampling ratio in (0, 1], ignored when --n is set")
    parser.add_argument("--n", type=int, default=0, help="Number of rows to keep, higher priority than --frac")
    return parser


def validate_args(args):
    """校验输入路径与采样参数是否合法。"""
    if not os.path.exists(args.input):
        raise FileNotFoundError(f"Input file not found: {args.input}")
    if args.n < 0:
        raise ValueError("--n must be >= 0")
    if not (0 < args.frac <= 1):
        raise ValueError("--frac must be in (0, 1]")


def count_lines(file_path):
    """统计文本文件总行数。"""
    total = 0
    with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
        for _ in f:
            total += 1
    return total


def resolve_target_lines(total_lines, n, frac):
    """根据行数或比例计算目标抽样条数。"""
    if total_lines <= 0:
        raise ValueError("Input file is empty")

    target = n if n > 0 else max(1, int(total_lines * frac))
    return min(target, total_lines)


def stream_copy_head(input_path, output_path, target_lines):
    """流式复制输入文件前 target_lines 行到输出文件。"""
    written = 0
    with open(input_path, "r", encoding="utf-8", errors="ignore") as fin:
        with open(output_path, "w", encoding="utf-8") as fout:
            for line in fin:
                if written >= target_lines:
                    break
                fout.write(line)
                written += 1
    return written


def main():
    """执行抽样主流程并输出结果文件。"""
    parser = build_parser()
    args = parser.parse_args()
    validate_args(args)

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)

    total_lines = count_lines(args.input)
    target_lines = resolve_target_lines(total_lines, args.n, args.frac)
    written = stream_copy_head(args.input, args.output, target_lines)

    print(f"Done. total={total_lines}, sampled={written}")
    print(f"Saved to: {args.output}")


if __name__ == "__main__":
    main()
