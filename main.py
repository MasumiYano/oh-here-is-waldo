from image_preprocessing import ImagePreprocessing
from utils import show_image, load_json, augment_image
import torch
import torch.optim as optim
from models import VGG16
from config import EPOCH

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

    model = VGG16(device, 2)
    optimizer = optim.Adam(model.parameters(), lr=0.0001, momentum=0.9)

    for epoch in range(EPOCH):
        total_losses = []
        total_accs = []
        for i, (name, tensor) in enumerate(train_data.items(), 0):
            inputs, labels = tesnor, get_name(name)


if __name__ == "__main__":
    main()
