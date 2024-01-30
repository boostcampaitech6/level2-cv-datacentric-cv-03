import os
import cv2
import glob
import multiprocessing
from tqdm import tqdm
import json
import numpy as np
from argparse import ArgumentParser
from concurrent.futures import ProcessPoolExecutor


def parse_args():
    parser = ArgumentParser()

    parser.add_argument(
        "--data_dir", type=str, default="./datasets/data/medical/img/test"
    )
    parser.add_argument(
        "--save_dir", type=str, default="./datasets/data/medical/img/myoutput"
    )
    parser.add_argument(
        "--output_csv", type=str, default="./predictions/output.csv"
    )

    args = parser.parse_args()

    return args


def multi_draw_boxes(img_paths, img_names, data, save_dir, n_cpu):
    assert len(img_paths) != 0, "no img file!"
    cnt = len(img_names)
    with ProcessPoolExecutor(max_workers=n_cpu - 1) as executor:
        list(
            tqdm(
                executor.map(
                    single_draw_boxes,
                    img_paths,
                    img_names,
                    [data] * cnt,
                    [save_dir] * cnt,
                ),
                total=len(img_names),
            )
        )


def single_draw_boxes(img_path, img_name, data, save_dir):
    img = cv2.imread(img_path)
    words = data["images"][img_name]["words"]

    for key in words.keys():
        points = data["images"][img_name]["words"][key]["points"]
        points = np.array(points, dtype=np.int32)
        cv2.polylines(
            img, [points], isClosed=True, color=(0, 0, 255), thickness=3
        )  # b g r

    cv2.imwrite(os.path.join(save_dir, img_name), img)


def main(args):
    data_dir = args["data_dir"]
    save_dir = args["save_dir"]
    output_csv = args["output_csv"]

    with open(output_csv, "r") as file:
        data = json.load(file)

    n_cpu = multiprocessing.cpu_count()
    os.makedirs(save_dir, exist_ok=True)

    img_paths = sorted(glob.glob(os.path.join(data_dir, "*.jpg")))
    img_names = sorted(os.listdir(data_dir))

    multi_draw_boxes(img_paths, img_names, data, save_dir, n_cpu)


if __name__ == "__main__":
    args = parse_args()
    main(vars(args))
