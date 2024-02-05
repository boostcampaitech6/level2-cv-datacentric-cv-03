"""

1. 예측을 잘 못하는 이미지의 bbox 정보를 받음
2. 그 bbox가 포함되는 1024x1024 Random Crop 이미지와 Json 파일 생성

"""

import numpy as np
import pandas as pd
import os
import json
import cv2

csv_path = "qr.csv"
root_path = "datasets/data/medical"
save_dir = f"{root_path}/img/train_crop"

json_path = f"{root_path}/ufo/train.json"
img_dir = f"{root_path}/img/train"

crop_size = 1024
ratio = 0.3
ignore_tags = ["masked", "excluded-region", "maintable", "stamp"]

print("start")
if not os.path.exists(save_dir):
    os.makedirs(save_dir, exist_ok=True)

df = pd.read_csv(csv_path, sep=",")

with open(json_path, "r") as r:
    json_data = json.load(r)

new_train_dict = {"images": {}}
for i in range(len(df)):
    img_name = df.iloc[i].images

    bbox_id = str(df.iloc[i].word_id).zfill(4)

    image_info = json_data["images"][f"{img_name}"]
    img_h, img_w = image_info["img_h"], image_info["img_w"]

    points = np.array(image_info["words"][f"{bbox_id}"]["points"])
    if len(points) > 4:
        print(f"{img_name} {bbox_id} : len(point) > 4")
        continue

    cx, cy = points[:, 0].mean(), points[:, 1].mean()
    x1, y1 = int(max(cx - crop_size / 2, 0)), int(max(cy - crop_size / 2, 0))
    x2, y2 = x1 + crop_size, y1 + crop_size

    tmp_mask = np.zeros((img_h, img_w), dtype=np.uint8)
    vertices = []
    new_words_dict = dict()
    for word_key, word_value in image_info["words"].items():
        word_tags = word_value["tags"]

        ignore_sample = any(elem for elem in word_tags if elem in ignore_tags)
        num_pts = np.array(word_value["points"]).shape[0]

        if ignore_sample or num_pts > 4:
            continue
        work_points = np.array(word_value["points"])

        tmp_area1 = tmp_mask.copy()
        cv2.fillPoly(tmp_area1, [work_points.astype(np.int32)], 1)

        tmp_area2 = tmp_mask.copy()
        new_points = work_points.copy()
        new_points[:, 0] -= x1
        new_points[:, 1] -= y1
        new_points = np.where(new_points < 0, 0, new_points)
        new_points = np.where(new_points > crop_size, crop_size, new_points)
        cv2.fillPoly(tmp_area2, [new_points.astype(np.int32)], 1)

        if tmp_area2.sum() / tmp_area1.sum() > ratio:
            word_value["points"] = new_points.tolist()
            new_words_dict[word_key] = word_value

    image_info["words"] = new_words_dict
    new_train_dict["images"][img_name] = image_info

    image = cv2.imread(os.path.join(img_dir, img_name))
    cv2.imwrite(os.path.join(save_dir, img_name), image[y1:y2, x1:x2, :])

with open(f"{root_path}/ufo/train_crop.json", "w") as w:
    json.dump(new_train_dict, w)

print("end")
