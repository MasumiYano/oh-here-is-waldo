from PIL import Image
import os
import matplotlib.pyplot as plt
import cv2
import json


def show_image(image_dict, json_data):
    fig, ax = plt.subplots(1, 4, figsize=(20, 5))  # 1 row, 4 columns
    image_counter = 0
    for file_name, img_arr in image_dict.items():
        if image_counter >= 4:
            break
        for data in json_data:
            if data.get("bg_name") == file_name:
                if data.get("class") == 0:
                    ax[image_counter].imshow(img_arr)
                    ax[image_counter].set_title(f'{file_name} (No Waldo)')
                else:
                    bbox = data.get("bbox")
                    img_with_bbox = cv2.rectangle(
                        img_arr.copy(),
                        (bbox[0], bbox[1]), 
                        (bbox[2], bbox[3]),
                        (255, 0, 0), 2
                    )
                    ax[image_counter].imshow(img_with_bbox)
                    ax[image_counter].set_title(f'{file_name} (Waldo)')
                ax[image_counter].axis('off')
                # Move to the next subplot
                image_counter += 1
                break

    plt.tight_layout()  # Adjust layout so the images don't overlap
    plt.show()


def load_json(json_file):
    with open(json_file, 'r') as file:
        data = json.load(file)
    return data
