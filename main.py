from image_preprocessing import ImagePreprocessing
from utils import show_image, load_json, augment_image
import torch

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


if __name__ == "__main__":
    main()
