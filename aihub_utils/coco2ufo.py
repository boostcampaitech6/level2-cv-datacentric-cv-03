# -*- coding: utf-8 -*-
import json

with open("/data/ephemeral/home/output.json", "rt") as f:
    data = f.read()

data = data.replace("}{", "},{")

data = "[" + data + "]"

input_json_array = json.loads(data)

output_json = {"images": {}}

for input_json in input_json_array:
    for image in input_json["images"]:
        image_dict = {"paragraphs": {}, "words": {}}

        for annotation in input_json["annotations"]:
            x, y, w, h = annotation["annotation.bbox"]
            new_points = [[x, y], [x + w, y], [x + w, y + h], [x, y + h]]
            annotation_dict = {
                "transcription": (annotation["annotation.text"]),
                "points": new_points,
                "orientation": "Horizontal",
                "language": None,
                "tags": ["inferred", "UpdatedTags", "occlusion"],
                "confidence": None,
                "illegibility": False,
            }

            image_dict["words"][str(annotation["id"])] = annotation_dict

        output_json["images"][image["image.file.name"]] = image_dict

output_json_string = json.dumps(output_json, indent=4)

# 저장 - 원하는 위치로 수정
with open("/data/ephemeral/home/data2/ufo/train.json", "wt") as f:
    f.write(output_json_string)
