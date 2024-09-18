from image_preprocessing import ImagePreprocessing
from utils import show_image, load_json, augment_image


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


if __name__ == "__main__":
    main()
