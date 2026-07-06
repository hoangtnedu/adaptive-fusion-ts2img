import argparse
from pathlib import Path
import shutil


def main() -> None:
    parser = argparse.ArgumentParser(description="Copy one UCR dataset from a source root to project data root.")
    parser.add_argument("--dataset", required=True, help="Dataset name, e.g. Coffee")
    parser.add_argument("--source-root", required=True, help="Root containing UCR folders, e.g. /content/drive/MyDrive/UCR")
    parser.add_argument("--target-root", default="data/UCR")
    args = parser.parse_args()

    src_dir = Path(args.source_root) / args.dataset
    dst_dir = Path(args.target_root) / args.dataset
    dst_dir.mkdir(parents=True, exist_ok=True)

    files = [
        f"{args.dataset}_TRAIN.tsv",
        f"{args.dataset}_TEST.tsv",
    ]

    for name in files:
        src = src_dir / name
        dst = dst_dir / name
        if not src.exists():
            raise FileNotFoundError(f"Source file not found: {src}")
        shutil.copy2(src, dst)
        print(f"Copied: {src} -> {dst}")

    print("Dataset copy completed:", dst_dir)


if __name__ == "__main__":
    main()
