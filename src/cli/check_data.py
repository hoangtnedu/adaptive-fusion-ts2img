import argparse
from pathlib import Path
import json

from src.data.ucr_loader import load_ucr_dataset
from src.utils.config import load_config


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate UCR dataset files and print basic statistics.")
    parser.add_argument("--config", default="config/default.yaml")
    parser.add_argument("--save-json", default=None, help="Optional path to save dataset statistics as JSON.")
    args = parser.parse_args()

    cfg = load_config(args.config)
    data_root = Path(cfg["paths"]["data_root"])
    dataset = cfg["experiment"]["dataset"]

    dataset_dir = data_root / dataset
    train_file = dataset_dir / f"{dataset}_TRAIN.tsv"
    test_file = dataset_dir / f"{dataset}_TEST.tsv"

    print("Config:", args.config)
    print("Dataset:", dataset)
    print("Train file:", train_file, "exists=", train_file.exists())
    print("Test file :", test_file, "exists=", test_file.exists())

    if not train_file.exists() or not test_file.exists():
        raise FileNotFoundError(
            "Dataset files not found. Expected files:\n"
            f"  {train_file}\n"
            f"  {test_file}\n"
        )

    X_train, y_train, X_test, y_test, label_encoder = load_ucr_dataset(data_root, dataset)

    stats = {
        "dataset": dataset,
        "train_file": str(train_file),
        "test_file": str(test_file),
        "n_train": int(X_train.shape[0]),
        "n_test": int(X_test.shape[0]),
        "series_length": int(X_train.shape[1]),
        "num_classes": int(len(label_encoder.classes_)),
        "classes": [str(c) for c in label_encoder.classes_],
        "x_train_shape": list(X_train.shape),
        "x_test_shape": list(X_test.shape),
    }

    print("\nDataset statistics")
    print(json.dumps(stats, indent=4, ensure_ascii=False))

    if args.save_json:
        out = Path(args.save_json)
        out.parent.mkdir(parents=True, exist_ok=True)
        with open(out, "w", encoding="utf-8") as f:
            json.dump(stats, f, indent=4, ensure_ascii=False)
        print("Saved:", out)


if __name__ == "__main__":
    main()
