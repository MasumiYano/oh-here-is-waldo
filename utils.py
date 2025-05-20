import random
from PIL import Image
import os
import matplotlib.pyplot as plt
import cv2
import json
import numpy as np
import torch
import torchvision.transforms as T

# Define transforms for data augmentation
transform = T.Compose([
    T.ToPILImage(),  # Convert numpy array to PIL Image first
    T.RandomHorizontalFlip(p=1.0),  # Always flip when augmentation is applied
    T.ColorJitter(hue=0.1, brightness=0.1, contrast=0.1, saturation=0.1),  # More conservative jitter
    T.ToTensor(),  # Convert to torch tensor
    T.ToPILImage(),  # Convert back to PIL for numpy conversion
])


def show_image(image_dict, json_data):
    """
    Display up to 4 images with their annotations.
    
    Args:
        image_dict: Dictionary of {filename: image_array}
        json_data: List of annotation dictionaries
    """
    fig, ax = plt.subplots(1, 4, figsize=(20, 5))  # 1 row, 4 columns
    image_counter = 0
    
    for file_name, img_arr in image_dict.items():
        if image_counter >= 4:
            break
            
        # Find corresponding annotation
        for data in json_data:
            if data.get("bg_name") == file_name:
                if data.get("class") == 0:
                    # No Waldo - just show image
                    ax[image_counter].imshow(img_arr)
                    ax[image_counter].set_title(f'{file_name} (No Waldo)')
                else:
                    # Waldo present - draw bounding box
                    bbox = data.get("bbox")
                    if bbox:
                        img_with_bbox = cv2.rectangle(
                            img_arr.copy(),
                            (bbox[0], bbox[1]),
                            (bbox[2], bbox[3]),
                            (255, 0, 0), 2
                        )
                        ax[image_counter].imshow(img_with_bbox)
                        ax[image_counter].set_title(f'{file_name} (Waldo)')
                    else:
                        ax[image_counter].imshow(img_arr)
                        ax[image_counter].set_title(f'{file_name} (Waldo - No BBox)')
                
                ax[image_counter].axis('off')
                image_counter += 1
                break

    # Hide unused subplots
    for i in range(image_counter, 4):
        ax[i].axis('off')

    plt.tight_layout()  # Adjust layout so the images don't overlap
    plt.show()  # Fixed: Added missing parentheses


def load_json(json_file):
    """
    Load JSON file and return data.
    
    Args:
        json_file: Path to JSON file
        
    Returns:
        data: Loaded JSON data
    """
    try:
        with open(json_file, 'r') as file:
            data = json.load(file)
        return data
    except FileNotFoundError:
        print(f"Warning: {json_file} not found. Returning empty list.")
        return []
    except json.JSONDecodeError:
        print(f"Warning: Error decoding {json_file}. Returning empty list.")
        return []


def update_bbox_after_flip(bbox, img_width):
    """
    Update bounding box coordinates after horizontal flip.
    
    Args:
        bbox: Original bounding box [x1, y1, x2, y2]
        img_width: Width of the image
        
    Returns:
        new_bbox: Updated bounding box coordinates
    """
    x1, y1, x2, y2 = bbox
    new_x1 = img_width - x2
    new_x2 = img_width - x1
    return [new_x1, y1, new_x2, y2]


def augment_image(image_dict, json_data):
    """
    Apply data augmentation to images and update annotations accordingly.
    
    Args:
        image_dict: Dictionary of {filename: image_array}
        json_data: List of annotation dictionaries
        
    Returns:
        shuffled_image_dict: Dictionary with original and augmented images
    """
    updated_json_data = json_data.copy()  # Copy original JSON data to preserve it
    new_images = {}  # Temporary dictionary to store new augmented images

    print("Applying data augmentation...")
    augmentation_count = 0

    for file_name, img_arr in image_dict.items():
        # Convert numpy array to PIL image for torchvision transforms
        if img_arr.dtype != np.uint8:
            img_arr = (img_arr * 255).astype(np.uint8)
        
        # With 50% probability, apply augmentation
        if np.random.rand() < 0.5:
            try:
                # Apply the augmentation
                augmented_img = transform(img_arr)
                augmented_img_arr = np.array(augmented_img)  # Convert back to NumPy

                # Create a new file name for the augmented image
                augmented_file_name = f"{file_name}_augmented"
                new_images[augmented_file_name] = augmented_img_arr  # Store in a separate dict
                augmentation_count += 1

                # Find the corresponding data in the JSON file
                for data in json_data:
                    if data.get("bg_name") == file_name:
                        # Copy the original data for the new augmented image entry
                        augmented_data = data.copy()
                        augmented_data["bg_name"] = augmented_file_name  # Update the file name for augmented image

                        # If the class is 1 (Waldo present), update the bounding box
                        if data.get("class") == 1 and data.get("bbox"):
                            bbox = data.get("bbox")
                            img_width = augmented_img_arr.shape[1]  # Width of the augmented image
                            updated_bbox = update_bbox_after_flip(bbox, img_width)  # Update bbox after flip
                            augmented_data['bbox'] = updated_bbox  # Update the bbox in the augmented data

                        # Add the augmented data to the updated JSON data
                        updated_json_data.append(augmented_data)
                        break  # No need to check further since we found the corresponding data
                        
            except Exception as e:
                print(f"Warning: Failed to augment {file_name}: {e}")
                continue

    print(f"Successfully augmented {augmentation_count} images")

    # After the iteration, update the original image_dict with the new images
    image_dict.update(new_images)

    # Shuffle the image_dict
    items = list(image_dict.items())  # Convert to list of tuples
    random.shuffle(items)  # Shuffle the list
    shuffled_image_dict = dict(items)  # Convert back to dictionary

    # Save the updated JSON file (with both original and augmented data)
    try:
        with open('updated_annotations.json', 'w') as json_file:
            json.dump(updated_json_data, json_file, indent=4)
        print("Updated annotations saved to 'updated_annotations.json'")
    except Exception as e:
        print(f"Warning: Failed to save updated annotations: {e}")

    return shuffled_image_dict  # Return the shuffled image dictionary


def get_name(img_name, json_data):
    """
    Get class and bounding box information for a given image name.
    
    Args:
        img_name: Image filename
        json_data: List of annotation dictionaries
        
    Returns:
        class_label: Class label (0 or 1)
        bbox_label: Bounding box coordinates [x1, y1, x2, y2] or [0, 0, 0, 0] if no Waldo
    """
    for data in json_data:
        if data.get("bg_name") == img_name:
            class_label = data.get("class", 0)  # Default to 0 if not found
            
            if class_label == 1:
                # Waldo present - return class and bbox
                bbox_label = data.get("bbox", [0, 0, 0, 0])  # Default bbox if missing
                return class_label, bbox_label
            else:
                # No Waldo - return class and default bbox
                return class_label, [0, 0, 0, 0]
    
    # If image not found in annotations, assume no Waldo
    print(f"Warning: No annotation found for {img_name}, assuming no Waldo")
    return 0, [0, 0, 0, 0]


def plot_graph(step_losses, step_accuracies):
    """
    Plot training loss and accuracy graphs.
    
    Args:
        step_losses: List of loss values
        step_accuracies: List of accuracy values
    """
    plt.figure(figsize=(12, 5))
    
    # Plot loss
    plt.subplot(1, 2, 1)
    plt.plot(step_losses, label="Training Loss", color='blue')
    plt.xlabel('Steps (x50)')
    plt.ylabel('Loss')
    plt.title("Training Loss")
    plt.legend()
    plt.grid(True, alpha=0.3)

    # Plot accuracy
    plt.subplot(1, 2, 2)
    plt.plot(step_accuracies, label="Training Accuracy", color='green')
    plt.xlabel('Steps (x50)')
    plt.ylabel('Accuracy')
    plt.title("Training Accuracy")
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()  # Fixed: Added missing parentheses


def visualize_predictions(model, test_tensors, test_labels, device, num_samples=4):
    """
    Visualize model predictions on test images.
    
    Args:
        model: Trained PyTorch model
        test_tensors: Dictionary of test image tensors
        test_labels: Dictionary of test labels
        device: Device to run inference on
        num_samples: Number of samples to visualize
    """
    model.eval()
    
    # Get random samples
    test_items = list(test_tensors.items())
    random.shuffle(test_items)
    
    fig, axes = plt.subplots(2, num_samples, figsize=(num_samples * 4, 8))
    if num_samples == 1:
        axes = axes.reshape(2, 1)
    
    with torch.no_grad():
        for idx, (name, tensor) in enumerate(test_items[:num_samples]):
            # Get prediction
            class_pred, bbox_pred = model(tensor)
            class_prob = torch.softmax(class_pred, dim=1)
            predicted_class = torch.argmax(class_prob, dim=1).item()
            confidence = class_prob[0, predicted_class].item()
            
            # Convert tensor back to image
            img = tensor.squeeze(0).cpu().permute(1, 2, 0).numpy()
            img = (img * 255).astype(np.uint8)
            
            # Get true labels
            true_class, true_bbox = test_labels[name]
            true_class = true_class.item() if isinstance(true_class, torch.Tensor) else true_class
            
            # Show original image with true bbox
            axes[0, idx].imshow(img)
            if true_class == 1 and any(true_bbox):
                bbox = true_bbox.cpu().numpy() if isinstance(true_bbox, torch.Tensor) else true_bbox
                rect = plt.Rectangle((bbox[0], bbox[1]), bbox[2]-bbox[0], bbox[3]-bbox[1], 
                                   fill=False, color='green', linewidth=2)
                axes[0, idx].add_patch(rect)
            axes[0, idx].set_title(f'True: {"Waldo" if true_class == 1 else "No Waldo"}')
            axes[0, idx].axis('off')
            
            # Show prediction
            axes[1, idx].imshow(img)
            if predicted_class == 1:
                bbox = bbox_pred.squeeze(0).cpu().numpy()
                rect = plt.Rectangle((bbox[0], bbox[1]), bbox[2]-bbox[0], bbox[3]-bbox[1], 
                                   fill=False, color='red', linewidth=2)
                axes[1, idx].add_patch(rect)
            axes[1, idx].set_title(f'Pred: {"Waldo" if predicted_class == 1 else "No Waldo"} ({confidence:.2f})')
            axes[1, idx].axis('off')
    
    plt.tight_layout()
    plt.show()
