from image_preprocessing import ImagePreprocessing
from utils import show_image, load_json


def main():
    preprocessing = ImagePreprocessing()
    preprocessing.resize_background()
    preprocessing.place_waldo_on_bg()
    preprocessing.convert_to_numpyarray()

    json_data = load_json("data/notation.json")
    show_image(preprocessing.waldo_and_backgrounds, json_data)

    for file_name, background in preprocessing.waldo_and_backgrounds.items():
        print(f"file name: {file_name}")


if __name__ == "__main__":
    main()
