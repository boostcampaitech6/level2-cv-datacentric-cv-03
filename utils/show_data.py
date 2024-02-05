import os
import json
from glob import glob
from pathlib import Path
from PIL import Image, ImageDraw


def read_json(filename: str):
    with Path(filename).open(encoding="utf8") as handle:
        ann = json.load(handle)
    return ann


def save_vis_to_img(
    save_dir: str | os.PathLike = None, img_lists: list = None
) -> None:
    if not os.path.exists(save_dir):
        os.makedirs(save_dir, exist_ok=True)

    ori_dict = {"Horizontal": "ho", "Vertical": "ve", "Irregular": "ir"}

    lag_dict = {
        None: "0",
        "others": "1",
        "ko": "2",
        "en": "3",
        "ko, others": "4",
        "ko, en": "5",
    }

    tag_dict = {
        "occlusion": "occ",
        "stamp": "stamp",
        "masked": "mask",
        "inferred": "infer",
    }

    for i in range(len(img_lists)):
        img_json = [
            [k, v] for k, v in data["images"].items() if k == img_lists[i]
        ]
        img_path = img_json[0][0]
        img = Image.open(
            os.path.join(
                "/opt/workspace/level2-cv-datacentric-cv-03/datasets/data/medical/img/train_crop/",
                img_path,
            )
        ).convert("RGB")
        draw = ImageDraw.Draw(img)

        # All of the prepared dataset consists of words. Not a character.
        for obj_k, obj_v in img_json[0][1]["words"].items():
            # language
            lan = None
            if isinstance(obj_v["language"], list):
                lan = ", ".join(obj_v["language"])
            else:
                lan = obj_v["language"]
            lan = lag_dict[lan]

            # orientation
            ori = ori_dict[obj_v["orientation"]]

            # tag (occlusion, stamp, masked, inferred)
            tag = None
            for t in obj_v["tags"]:
                try:
                    tag += tag_dict[t]
                except:
                    pass

            if tag is None:
                obj_name = f"{ori}_{obj_k}_{lan}"
            else:
                obj_name = f"{tag}_{ori}_{obj_k}_{lan}"

            # bbox points
            pts = [(int(p[0]), int(p[1])) for p in obj_v["points"]]
            pt1 = sorted(pts, key=lambda x: (x[1], x[0]))[0]

            # Masking object which not use for training.

            if obj_v["illegibility"]:
                # draw.polygon(pts, fill=(0, 0, 0))
                pass
            else:
                draw.polygon(pts, outline=(255, 0, 0))
                draw.text((pt1[0] - 3, pt1[1] - 12), obj_name, fill=(0, 0, 0))
        img.save(os.path.join(save_dir, img_path))


if __name__ == "__main__":
    data = read_json("./datasets/data/medical/ufo/train.json")
    img_lists = os.listdir("./datasets/data/medical/img/train")
    save_vis_to_img("./datasets/data/medical/img/res_vis", img_lists)
