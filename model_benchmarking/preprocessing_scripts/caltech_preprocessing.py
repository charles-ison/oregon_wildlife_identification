import json
import os
from PIL import Image
from datetime import datetime

file_name = "animal_count_key.json"
json_file = open(file_name)
coco_key = json.load(json_file)
annotations = coco_key["annotations"]
images = coco_key["images"]

for json_image in images:
    json_image["file_name"] = "images/" + json_image["file_name"]
    time_stamp = datetime.strptime(json_image["date_captured"], '%Y-%m-%d %H:%M:%S')
    date_time_string = datetime.strftime(time_stamp, "%Y:%m:%d %H:%M:%S")
    json_image["datetime"] = date_time_string

for json_annotation in annotations:
    json_annotation["category_id"] = 1 if json_annotation["category_id"] == 34 else 0

json_file.close()
json_file = open(file_name, "w")
json.dump(coco_key, json_file, indent = 4)
json_file.close()