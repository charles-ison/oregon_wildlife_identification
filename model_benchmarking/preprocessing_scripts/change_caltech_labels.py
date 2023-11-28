import json
import os
from PIL import Image

file_name = "animal_count_key.json"
json_file = open(file_name)
coco_key = json.load(json_file)
annotations = coco_key["annotations"]

for json_annotation in annotations:
    json_annotation["category_id"] = 1 if json_annotation["category_id"] == 34 else 0

json_file.close()
json_file = open(file_name, "w")
json.dump(coco_key, json_file, indent = 4)
json_file.close()