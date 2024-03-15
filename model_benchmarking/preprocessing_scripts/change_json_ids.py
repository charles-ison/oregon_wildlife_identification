import json
import os
from PIL import Image

file_name = "instances_default.json"
json_file = open(file_name)
coco_key = json.load(json_file)
images = coco_key["images"]
annotations = coco_key["annotations"]
new_starting_image_id = 290
new_starting_annotation_id = 60
image_id_dict = {};

for json_image in images:
    image_id_dict[json_image["id"]] = new_starting_image_id
    json_image["id"] = new_starting_image_id
    new_starting_image_id += 1

for json_annotation in annotations:
    new_image_id = image_id_dict[json_annotation["image_id"]]
    json_annotation["image_id"] = new_image_id
    json_annotation["id"] = new_starting_annotation_id
    new_starting_annotation_id += 1

json_file.close()

json_file = open(file_name, "w")
json.dump(coco_key, json_file, indent = 4)
json_file.close()