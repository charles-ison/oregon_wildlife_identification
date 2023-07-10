import json
import os
from PIL import Image

json_file = open("animal_count_key.json")
coco_key = json.load(json_file)
images = coco_key["images"]

for index, json_image in enumerate(images):
    file_name = json_image["file_name"]
    if os.path.isfile(file_name):
        image = Image.open(file_name)
        json_image["datetime"] = image._getexif()[36867]
json_file.close()

json_file = open("animal_count_key.json", "w")
json.dump(coco_key, json_file, indent = 4)
json_file.close()