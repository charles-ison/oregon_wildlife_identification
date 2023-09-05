import json
import os
from PIL import Image

file_name = "instances_default.json"
json_file = open(file_name)
coco_key = json.load(json_file)
images = coco_key["images"]

for json_image in images:
    json_image["file_name"] = "6.29-7.06/" + json_image["file_name"]
json_file.close()

json_file = open(file_name, "w")
json.dump(coco_key, json_file, indent = 4)
json_file.close()