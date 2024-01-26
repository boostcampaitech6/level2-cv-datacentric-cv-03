import json
import os
from sklearn.model_selection import KFold
from argparse import ArgumentParser

SEED = 42


def parse_args():
    parser = ArgumentParser()

    # Conventional args
    parser.add_argument(
        "--anno_dir",
        "-a",
        type=str,
        default="./datasets/data/medical/ufo/train.json",
    )
    parser.add_argument(
        "--save_dir",
        "-s",
        type=str,
        default="./datasets/data/medical/ufo/split",
    )

    args = parser.parse_args()

    return args


def main(args):
    annotation = args["anno_dir"]
    save_dir = args["save_dir"]

    # make folder
    os.makedirs(save_dir)

    # train.json 파일 읽기
    with open(annotation, "r") as file:
        data = json.load(file)

    # 'images' 키에서 이미지 데이터 추출
    image_files = [key for key in data["images"].keys()]

    # 5-Fold로 데이터 분할
    kf = KFold(n_splits=5, shuffle=True, random_state=SEED)

    for fold, (train_idx, val_idx) in enumerate(
        kf.split(image_files), start=1
    ):
        train_data = [image_files[i] for i in train_idx]
        val_data = [image_files[i] for i in val_idx]

        train_json, val_json = {"images": {}}, {"images": {}}

        for image in train_data:
            train_json["images"][image] = data["images"][image]
        for image in val_data:
            val_json["images"][image] = data["images"][image]

        save_train_path = os.path.join(
            save_dir, f"train_{SEED}_fold_{fold}.json"
        )
        save_val_path = os.path.join(save_dir, f"val_{SEED}_fold_{fold}.json")

        with open(save_train_path, "w") as f:
            json.dump(train_json, f, indent=4)

        with open(save_val_path, "w") as f:
            json.dump(val_json, f, indent=4)

        print(f"Fold {fold} saved to {save_train_path} and {save_val_path}")


if __name__ == "__main__":
    args = parse_args()
    main(vars(args))
