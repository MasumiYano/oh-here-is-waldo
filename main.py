from image_preprocessing import ImagePreprocessing
from utils import show_image, load_json, augment_image, get_name, plot_graph
import torch
import torch.optim as optim
from models import VGG16, compute_loss
from config import EPOCH
import torch.nn as nn

device = "cuda" if torch.cuda.is_available() else "cpu"


def main():
    preprocessing = ImagePreprocessing()
    preprocessing.resize_background()
    preprocessing.place_waldo_on_bg()
    preprocessing.convert_to_numpyarray()

    json_data = load_json("data/notation.json")
    show_image(preprocessing.waldo_and_backgrounds, json_data)

    augmented_dict = augment_image(preprocessing.waldo_and_backgrounds, json_data)
    augmented_json = load_json("updated_annotations.json")
    show_image(augmented_dict, augmented_json)

    split_idx = int(len(augmented_dict) * 0.7)
    items = list(augmented_dict.items())
    train_data, test_data = dict(items[:split_idx]), dict(items[split_idx:])

    model = VGG16(device, 2).to(device)
    optimizer = optim.adam(model.parameters(), lr=0.0001, weight_decay=0.01)

    step_losses = []
    step_accuraceis = []
    step_interval = 50

    model.train()
    for epoch in range(EPOCH):
        total_loss = 0.0
        total_accuracy = 0.0
        step_count = 0

        for i, (name, tensor) in enumerate(train_data.items(), 0):
            inputs = tensor.to(device)
            class_true, bbox_true = get_name(name, augmented_json)
            class_true, bbox_true = class_true.to(device), bbox_true.to(device)

            optimizer.zero_grad()

            class_pred, bbox_pred = model(inputs)
            loss = compute_loss(class_pred, class_true, bbox_pred, bbox_true)
            loss.backward()
            optimizer.step()


            total_loss += loss.item()

            accuracy = calculate_accuracy(class_pred, class_true)
            total_accuracy += accuracy
            step_count += 1

            if step_count % step_interval == 0:
                avg_loss = total_loss / step_count
                avg_accuracy = total_accuracy / step_count
                step_losses.append(avg_loss)
                step_accuraceis.append(avg_accuracy)
                print(f"Epoch [{epoch+1}/{EPOCH}], Step [{step_count}, Loss: {avg_loss:.4f}, Accuracy: {avg_accuracy:.4f}]")

        print(f"Epoch [{epoch}/{EPOCH}], Final Loss: {total_loss/len(train_data)}, Final Accuracy: {total_accuracy/len(train_data)}")

        plot_graph(step_losses, step_accuraceis)


if __name__ == "__main__":
    main()
