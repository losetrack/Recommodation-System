import argparse
import os


def build_parser():
    parser = argparse.ArgumentParser(description="Split a large train.txt into streaming train/val files.")
    parser.add_argument("--input", type=str, default="./data/train.txt", help="Path to source train file")
    parser.add_argument("--train-output", type=str, default="./data/train_stream.txt", help="Output path for train split")
    parser.add_argument("--val-output", type=str, default="./data/val_stream.txt", help="Output path for validation split")
    parser.add_argument("--val-ratio", type=float, default=0.1, help="Validation ratio in (0, 1)")
    return parser


def validate_args(args):
    if not os.path.exists(args.input):
        raise FileNotFoundError(f"Input file not found: {args.input}")
    if not 0 < args.val_ratio < 1:
        raise ValueError(f"--val-ratio must be between 0 and 1, got {args.val_ratio}")


def count_lines(file_path):
    total = 0
    with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
        for _ in f:
            total += 1
    if total < 2:
        raise ValueError("Input file must contain at least 2 lines")
    return total


def split_file(input_path, train_output_path, val_output_path, train_lines):
    written_train = 0
    written_val = 0

    with open(input_path, "r", encoding="utf-8", errors="ignore") as fin:
        with open(train_output_path, "w", encoding="utf-8") as train_out:
            with open(val_output_path, "w", encoding="utf-8") as val_out:
                for idx, line in enumerate(fin):
                    if idx < train_lines:
                        train_out.write(line)
                        written_train += 1
                    else:
                        val_out.write(line)
                        written_val += 1

    return written_train, written_val


def main():
    parser = build_parser()
    args = parser.parse_args()
    validate_args(args)

    os.makedirs(os.path.dirname(args.train_output) or ".", exist_ok=True)
    os.makedirs(os.path.dirname(args.val_output) or ".", exist_ok=True)

    print("Counting lines...")
    total_lines = count_lines(args.input)
    val_lines = max(1, int(total_lines * args.val_ratio))
    train_lines = total_lines - val_lines

    print(
        f"Splitting {total_lines} lines into "
        f"{train_lines} train / {val_lines} val..."
    )
    written_train, written_val = split_file(
        args.input,
        args.train_output,
        args.val_output,
        train_lines,
    )

    print(f"Done. train={written_train}, val={written_val}")
    print(f"Train split: {args.train_output}")
    print(f"Val split:   {args.val_output}")


if __name__ == "__main__":
    main()
