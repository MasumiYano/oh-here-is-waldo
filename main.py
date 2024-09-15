from image_preprocessing import ImagePreprocessing


def main():
    preprocessing = ImagePreprocessing()
    preprocessing.resize_background()
    preprocessing.place_waldo_on_bg()
    preprocessing.convert_to_numpyarray()

    print(preprocessing.waldo_and_backgrounds[:3])

if __name__ == "__main__":
    main()
