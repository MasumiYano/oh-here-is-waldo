from PIL import Image
import os
import matplotlib.pyplot as plt
import cv2
import json


def show_image(image_dict, json_data):
    for file_name, img_arr in image_dict.items():
        for data in json_data:
            if data.get("bg_name") == file_name:
                if data.get("class") == 0:
                    plt.imshow(img_arr)
                    plt.show()
                else:
                    bbox = data.get("bbox")
                    img_with_bbox = cv2.rectangle(img_arr.copy(),
                                                  (bbox[0], bbox[1]),
                                                  (bbox[2], bbox[3],
                                                   (255, 0, 0), 2))
                    plt.imshow(img_with_bbox)
                    plt.show()


def load_json(json_file):
    with open(json_file, 'r') as file:
        data = json.load(file)
    return data
