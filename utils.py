import random
from PIL import Image
import os
import matplotlib.pyplot as plt
import cv2
import json
import numpy as np
import torchvision.transforms as T

transform = T.Compose([
    T.RandomHorizontalFlip(p=1),  # Horizontal flip with 50% chance
    T.ColorJitter(hue=0.5),          # Random hue adjustment
    T.ToTensor()  # convert to torch tensor
])


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


def update_bbox_after_flip(bbox, img_width):
    x1, y1, x2, y2 = bbox
    new_x1 = img_width - x2
    new_x2 = img_width - x1
    return [new_x1, y1, new_x2, y2]


def update_bbox_after_flip(bbox, img_width):
    x1, y1, x2, y2 = bbox
    new_x1 = img_width - x2
    new_x2 = img_width - x1
    return [new_x1, y1, new_x2, y2]


def augment_image(image_dict, json_data):
    updated_json_data = json_data.copy()  # Copy original JSON data to preserve it
    new_images = {}  # Temporary dictionary to store new augmented images

    for file_name, img_arr in image_dict.items():
        img = Image.fromarray(img_arr)  # Convert NumPy array to PIL image

        # With 50% probability, apply augmentation
        if np.random.rand() < 0.5:
            # Apply the augmentation
            augmented_img = transform(img)
            augmented_img_arr = np.array(augmented_img)  # Convert back to NumPy

            # Create a new file name for the augmented image
            augmented_file_name = f"{file_name}_augmented"
            new_images[augmented_file_name] = augmented_img_arr  # Store in a separate dict

            # Find the corresponding data in the JSON file
            for data in json_data:
                if data.get("bg_name") == file_name:
                    # Copy the original data for the new augmented image entry
                    augmented_data = data.copy()
                    augmented_data["bg_name"] = augmented_file_name  # Update the file name for augmented image

                    # If the class is 1, update the bounding box
                    if data.get("class") == 1:
                        bbox = data.get("bbox")
                        img_width = augmented_img_arr.shape[1]  # Width of the augmented image
                        bbox = update_bbox_after_flip(bbox, img_width)  # Update bbox after flip
                        augmented_data['bbox'] = bbox  # Update the bbox in the augmented data

                    # Add the augmented data to the updated JSON data
                    updated_json_data.append(augmented_data)
                    break  # No need to check further since we found the corresponding data

    # After the iteration, update the original image_dict with the new images
    image_dict.update(new_images)

    # Shuffle the image_dict
    items = list(image_dict.items())  # Convert to list of tuples
    random.shuffle(items)  # Shuffle the list
    shuffled_image_dict = dict(items)  # Convert back to dictionary

    # Save the updated JSON file (with both original and augmented data)
    with open('updated_annotations.json', 'w') as json_file:
        json.dump(updated_json_data, json_file, indent=4)

    return shuffled_image_dict  # Return the shuffled image dictionary


def get_name(img_name, json):
    for data in json:
        if data.get("bg_name") == img_name:
            if data.get("class") == 1:
                return data.get("class"), data.get("bbox")
            else:
                return data.get("class"), [0, 0, 0, 0]


def plot_graph(step_losses, step_acc):
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.plot(step_losses, label="Loss")
    plt.xlabel('Steps (x50)')
    plt.ylabel('Loss')
    plt.title("Training Loss")
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(step_acc, label="Accuracy")
    plt.xlabel('Steps (x50)')
    plt.ylabel('Accuracy')
    plt.title("Training Accuracy")
    plt.legend()

    plt.tight_layout()
    plt.show

