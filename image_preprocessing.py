from PIL import Image
import glob
import os
import random
import json

from config import (IMAGE_WIDTH, IMAGE_HEIGHT)


class ImagePreprocessing:
    def __init__(self):
        self.image_width = IMAGE_WIDTH
        self.image_height = IMAGE_HEIGHT
        self.waldo_path = "data/waldo.png"
        self.waldo = Image.open(self.waldo_path)

        waldo_width, waldo_height = self.waldo.size
        if waldo_width > self.image_width or waldo_height > self.image_height:
            factor = min(
                (self.image_width * 0.2) / waldo_width,
                (self.image_height * 0.2) / waldo_height
            )
            self.waldo = self.waldo.resize(
                (int(waldo_width * factor), int(waldo_height * factor)),
                Image.LANCZOS
            )

        self.background_paths = glob.glob('data/images/*.jpg') + glob.glob('data/images/*.jpeg')
        self.resized_backgrounds = []
        self.waldo_and_backgrounds = []

    def resize_background(self):
        for background in self.background_paths:
            bg_img = Image.open(background)
            bg_width, bg_height = bg_img.size

            if bg_width != self.image_width or bg_height != self.image_height:
                resized_bg = bg_img.resize(
                    (self.image_width, self.image_height),
                    Image.LANCZOS
                )
            else:
                resized_bg = bg_img.copy()

            self.resized_backgrounds.append((resized_bg, background))

    def place_waldo_on_bg(self):
        waldo_width, waldo_height = self.waldo.size
        for bg_img, bg_filename in self.resized_backgrounds:
            bg = bg_img.convert("RGBA")
            max_width_pos = max(0, self.image_width - waldo_width)
            max_height_pos = max(0, self.image_height - waldo_height)
            x_pos = random.randint(0, max_width_pos)
            y_pos = random.randint(0, max_height_pos)
            include = 0

            if random.random() < 0.7:
                include = 1
                bg.paste(self.waldo, (x_pos, y_pos), self.waldo)

            self.mark_waldo_pos(x_pos, y_pos, waldo_width, waldo_height, bg_filename, include)
            self.waldo_and_backgrounds.append(bg)

    def mark_waldo_pos(self, x_pos, y_pos, waldo_width, waldo_height, bg_filename, include):
        file_path = os.path.join(os.getcwd(), "data", "notation.json")

        data_dir = os.path.join(os.getcwd(), "data")
        os.makedirs(data_dir, exist_ok=True)

        data = {
            "bg_name": os.path.basename(bg_filename),
            "is_waldo": include
        }

        if include == 1:
            data.update({
                "x_start": x_pos,
                "y_start": y_pos,
                "x_end": x_pos + waldo_width,
                "y_end": y_pos + waldo_height
            })

        if os.path.exists(file_path):
            with open(file_path, 'r') as file:
                try:
                    existing_data = json.load(file)
                except json.JSONDecodeError:
                    print("Oops, JSONDecodeError occurred.")
                    existing_data = []
        else:
            existing_data = []

        existing_data.append(data)

        with open(file_path, 'w') as file:
            json.dump(existing_data, file, indent=4)
