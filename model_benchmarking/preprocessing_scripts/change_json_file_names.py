import json
import os
from PIL import Image

json_file = open("8.03-8.12.json")
coco_key = json.load(json_file)
images = coco_key["images"]

for json_image in images:
    json_image["file_name"] = "8.03-8.12/" + json_image["file_name"]
json_file.close()

json_file = open("8.03-8.12.json", "w")
json.dump(coco_key, json_file, indent = 4)
json_file.close()