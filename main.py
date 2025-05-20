from image_preprocessing import ImagePreprocessing
from utils import show_image, load_json, augment_image, get_name, plot_graph
import torch
import torch.optim as optim
from models import VGG16, compute_loss
from config import EPOCH
import torch.nn as nn
import numpy as np

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")


def calculate_accuracy(class_pred, class_true):
    """
    Calculate classification accuracy.
    
    Args:
        class_pred: Predicted class logits [batch_size, num_classes]
        class_true: True class labels [batch_size]
    
    Returns:
        accuracy: Classification accuracy as a float
    """
    # Get predicted classes by taking argmax
    predicted_classes = torch.argmax(class_pred, dim=1)
    
    # Calculate accuracy
    correct = (predicted_classes == class_true).float()
    accuracy = correct.mean().item()
    
    return accuracy


def convert_to_tensor(image_dict, json_data, device):
    """
    Convert image dictionary to tensors and prepare labels.
    
    Args:
        image_dict: Dictionary of images {filename: numpy_array}
        json_data: List of annotation dictionaries
        device: Device to move tensors to
    
    Returns:
        tensor_dict: Dictionary of {filename: tensor}
        labels_dict: Dictionary of {filename: (class_tensor, bbox_tensor)}
    """
    tensor_dict = {}
    labels_dict = {}
    
    for filename, img_array in image_dict.items():
        # Convert image to tensor and normalize
        img_tensor = torch.from_numpy(img_array).float() / 255.0
        
        # Handle different array shapes (H, W, C) -> (C, H, W)
        if len(img_tensor.shape) == 3:
            img_tensor = img_tensor.permute(2, 0, 1)
        
        # Add batch dimension and move to device
        img_tensor = img_tensor.unsqueeze(0).to(device)
        tensor_dict[filename] = img_tensor
        
        # Get labels
        class_label, bbox_label = get_name(filename, json_data)
        
        # Convert labels to tensors
        if isinstance(class_label, (list, tuple)):
            class_tensor = torch.tensor(class_label[0]).long().to(device)
        else:
            class_tensor = torch.tensor(class_label).long().to(device)
            
        if isinstance(bbox_label, (list, tuple)):
            bbox_tensor = torch.tensor(bbox_label).float().to(device)
        else:
            bbox_tensor = torch.tensor([0, 0, 0, 0]).float().to(device)
            
        labels_dict[filename] = (class_tensor, bbox_tensor)
    
    return tensor_dict, labels_dict


def main():
    # Data preprocessing
    print("Starting image preprocessing...")
    preprocessing = ImagePreprocessing()
    preprocessing.resize_background()
    preprocessing.place_waldo_on_bg()
    preprocessing.convert_to_numpyarray()

    # Load annotations
    json_data = load_json("data/notation.json")
    print(f"Loaded {len(json_data)} annotations")
    
    # Show original images
    show_image(preprocessing.waldo_and_backgrounds, json_data)

    # Augment images
    print("Augmenting images...")
    augmented_dict = augment_image(preprocessing.waldo_and_backgrounds, json_data)
    augmented_json = load_json("updated_annotations.json")
    print(f"Total images after augmentation: {len(augmented_dict)}")
    
    # Show augmented images
    show_image(augmented_dict, augmented_json)

    # Split data into train and test
    split_idx = int(len(augmented_dict) * 0.7)
    items = list(augmented_dict.items())
    train_data, test_data = dict(items[:split_idx]), dict(items[split_idx:])
    print(f"Training samples: {len(train_data)}, Test samples: {len(test_data)}")

    # Convert to tensors
    print("Converting data to tensors...")
    train_tensors, train_labels = convert_to_tensor(train_data, augmented_json, device)
    test_tensors, test_labels = convert_to_tensor(test_data, augmented_json, device)

    # Initialize model
    print("Initializing model...")
    model = VGG16(device, 2).to(device)
    
    # Fix: Use correct optimizer initialization
    optimizer = optim.Adam(model.parameters(), lr=0.0001, weight_decay=0.01)

    # Training tracking
    step_losses = []
    step_accuracies = []  # Fix: correct spelling
    step_interval = 50

    print(f"Starting training for {EPOCH} epochs...")
    model.train()
    
    for epoch in range(EPOCH):
        total_loss = 0.0
        total_accuracy = 0.0
        step_count = 0

        # Shuffle training data each epoch
        train_items = list(train_tensors.items())
        np.random.shuffle(train_items)

        for i, (name, tensor) in enumerate(train_items):
            # Get inputs and labels
            inputs = tensor
            class_true, bbox_true = train_labels[name]
            
            # Add batch dimension if not present
            if class_true.dim() == 0:
                class_true = class_true.unsqueeze(0)
            if bbox_true.dim() == 1:
                bbox_true = bbox_true.unsqueeze(0)

            # Zero gradients
            optimizer.zero_grad()

            # Forward pass
            class_pred, bbox_pred = model(inputs)
            
            # Calculate loss
            loss = compute_loss(class_pred, class_true, bbox_pred, bbox_true)
            
            # Backward pass
            loss.backward()
            optimizer.step()

            # Update tracking
            total_loss += loss.item()
            accuracy = calculate_accuracy(class_pred, class_true)
            total_accuracy += accuracy
            step_count += 1

            # Log progress
            if step_count % step_interval == 0:
                avg_loss = total_loss / step_count
                avg_accuracy = total_accuracy / step_count
                step_losses.append(avg_loss)
                step_accuracies.append(avg_accuracy)
                print(f"Epoch [{epoch+1}/{EPOCH}], Step [{step_count}], Loss: {avg_loss:.4f}, Accuracy: {avg_accuracy:.4f}")

        # Epoch summary
        avg_epoch_loss = total_loss / len(train_items)
        avg_epoch_accuracy = total_accuracy / len(train_items)
        print(f"Epoch [{epoch+1}/{EPOCH}] - Final Loss: {avg_epoch_loss:.4f}, Final Accuracy: {avg_epoch_accuracy:.4f}")

        # Validation (optional - run on test set)
        if epoch % 5 == 0:  # Validate every 5 epochs
            model.eval()
            val_loss = 0.0
            val_accuracy = 0.0
            
            with torch.no_grad():
                for name, tensor in test_tensors.items():
                    inputs = tensor
                    class_true, bbox_true = test_labels[name]
                    
                    if class_true.dim() == 0:
                        class_true = class_true.unsqueeze(0)
                    if bbox_true.dim() == 1:
                        bbox_true = bbox_true.unsqueeze(0)
                    
                    class_pred, bbox_pred = model(inputs)
                    loss = compute_loss(class_pred, class_true, bbox_pred, bbox_true)
                    accuracy = calculate_accuracy(class_pred, class_true)
                    
                    val_loss += loss.item()
                    val_accuracy += accuracy
            
            avg_val_loss = val_loss / len(test_tensors)
            avg_val_accuracy = val_accuracy / len(test_tensors)
            print(f"Validation - Loss: {avg_val_loss:.4f}, Accuracy: {avg_val_accuracy:.4f}")
            
            model.train()  # Back to training mode

    # Plot training progress
    print("Plotting training progress...")
    plot_graph(step_losses, step_accuracies)
    
    # Save model
    print("Saving model...")
    torch.save(model.state_dict(), 'waldo_model.pth')
    print("Training completed successfully!")


if __name__ == "__main__":
    main()
