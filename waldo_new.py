import os
import random
from PIL import Image
import matplotlib.pyplot as plt
import glob
import cv2
import json
import numpy as np
from sklearn.model_selection import train_test_split
import shutil
import albumentations as alb
from pathlib import Path
import tensorflow as tf

# %% md
## Randomly place waldo on backgrounds
# %%
IMAGE_WIDTH = 800
IMAGE_HEIGHT = 600

SCALE_DOWN_WIDTH = 170
SCALE_DOWN_HEIGHT = 170


# %%
def resize_background(bg_path, target_width, target_height):
    with Image.open(bg_path) as bg:
        bg_width, bg_height = bg.size
        #         Enlarge
        if bg_width < target_width or bg_height < target_height:
            resized_bg = bg.resize((target_width, target_height), Image.LANCZOS)
        #             Reduce
        elif bg_width > target_width or bg_height > target_height:
            resized_bg = bg.resize((target_width, target_height), Image.LANCZOS)
        else:
            resized_bg = bg.copy()  # Copy to ensure we don't return a closed image
        return resized_bg


# %%
def place_waldo_on_bg(waldo_path, bg_image):
    # Ensure the background is in RGBA
    bg_image = bg_image.convert("RGBA")

    # Open the Waldo image (in PNG format)
    with Image.open(waldo_path) as waldo:
        waldo = waldo.convert("RGBA")  # Ensure Waldo is in RGBA mode
        waldo_width, waldo_height = waldo.size
        bg_width, bg_height = bg_image.size

        # Scale Waldo
        if waldo_width > bg_width or waldo_height > bg_height:
            factor = min((bg_width * .2) / waldo_width, (bg_height * .2) / waldo_height)
            waldo = waldo.resize((int(waldo_width * factor), int(waldo_height * factor)), Image.LANCZOS)
            waldo_width, waldo_height = waldo.size

        # Random location for Waldo
        max_width_position = bg_width - waldo_width
        max_height_position = bg_height - waldo_height
        position_width = random.randint(0, max_width_position)
        position_height = random.randint(0, max_height_position)

        bg_image.paste(waldo, (position_width, position_height), waldo)

        # Convert the result back to RGB to save in JPEG format
        bg_image = bg_image.convert("RGB")

        return bg_image


# %%
def send_img_to_directory(directory_path, image, file_name):
    if not os.path.exists(directory_path):
        os.makedirs(directory_path)

    # Check if the image has an alpha channel (RGBA)
    if image.mode == 'RGBA':
        # Create a white RGB background
        background = Image.new('RGB', image.size, (255, 255, 255))
        # Paste using alpha channel as mask
        background.paste(image, (0, 0), image)
        image = background

    file_path = os.path.join(directory_path, file_name)
    image.save(file_path, 'JPEG')


# %%
waldo_path = 'data/images/waldo.png'
waldo = Image.open(waldo_path)
background_paths = glob.glob('data/images/*.jpg') + glob.glob('data/images/*.jpeg')

for i, bg_path in enumerate(background_paths):
    resized_bg = resize_background(bg_path=bg_path, target_width=IMAGE_WIDTH, target_height=IMAGE_HEIGHT)
    background_with_waldo = place_waldo_on_bg(waldo_path=waldo_path, bg_image=resized_bg)
    file_name = f'bg{i + 1}_with_waldo.jpg'
    send_img_to_directory(directory_path='data/created_images', image=background_with_waldo, file_name=file_name)
# %% md
## Label the waldo for each generated picture using labelme
# %%
# !labelme
# %% md
## Make the image file into array
# %%
images = tf.data.Dataset.list_files('data/created_images/*.jpg', shuffle=False)


# %%
def load_image(file_path):
    byte_img = tf.io.read_file(file_path)  # byte coded image
    img = tf.io.decode_jpeg(byte_img)  # decote it
    return img


# %%
images = images.map(load_image)  # apply load image function for each value in the dataset.
# %%
# batch returns the number of the parameter. In this case, it returns 4 images.
image_generator = images.batch(4).as_numpy_iterator()
# %%
plot_images = image_generator.next()
# %%
fig, ax = plt.subplots(ncols=4, figsize=(20, 20))
for idx, image in enumerate(plot_images):
    ax[idx].imshow(image)
plt.show()
# %% md
## Splitting image and json file into train, test and validation
# %%
images_dir = 'data/created_images'
json_dir = 'data/processed_img_info'

image_paths = []
json_paths = []

# Load Images and corresponding Json file.
for image_file in os.listdir(images_dir):
    if image_file.endswith('.jpg'):
        image_path = os.path.join(images_dir, image_file)
        json_file = image_file.replace('.jpg', '.json')
        json_path = os.path.join(json_dir, json_file)

        if os.path.exists(json_path):
            image_paths.append(image_path)
            json_paths.append(json_path)

# Split data
train_img_paths, test_val_img_paths, train_json_paths, test_val_json_paths = train_test_split(image_paths, json_paths,
                                                                                              test_size=0.3,
                                                                                              random_state=42)
test_img_paths, val_img_paths, test_json_path, val_json_path = train_test_split(test_val_img_paths, test_val_json_paths,
                                                                                test_size=0.5, random_state=42)


# Copy files into respective directories
def copy_files(image_paths, json_paths, target_img_dir, target_json_dir):
    os.makedirs(target_img_dir, exist_ok=True)
    os.makedirs(target_json_dir, exist_ok=True)

    for img_path, json_path in zip(image_paths, json_paths):
        shutil.copy(img_path, target_img_dir)
        shutil.copy(json_path, target_json_dir)


# %%
copy_files(train_img_paths, train_json_paths, 'data/train/images', 'data/train/labels')
copy_files(test_img_paths, test_json_path, 'data/test/images', 'data/test/labels')
copy_files(val_img_paths, val_json_path, 'data/validation/images', 'data/validation/labels')
# %% md
## Augment images to increase the size of train, test, and validation dataset.
# %%
augmentor = alb.Compose([alb.RandomCrop(width=450, height=450),
                         alb.HorizontalFlip(p=0.5),
                         alb.RandomBrightnessContrast(p=0.2),
                         alb.RandomGamma(p=0.2),
                         alb.RGBShift(p=0.2),
                         alb.VerticalFlip(p=0.2)],
                        bbox_params=alb.BboxParams(format='albumentations', label_fields=['class_labels']))


# %%
def load_label(label_path):
    if label_path.exists():
        with label_path.open('r') as file:
            label = json.load(file)
            coords = [coord for point in label['shapes'][0]['points'] for coord in point]
            return list(np.divide(coords, [IMAGE_WIDTH, IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_HEIGHT]))


# %%
def process_image(image_path, label_path, partition):
    img = cv2.imread(str(image_path))
    coords = load_label(label_path)

    for x in range(60):
        augmented = augmentor(image=img, bboxes=[coords], class_labels=['waldo'])
        augmented_image_path = Path('aug_data', partition, 'images', f'{image_path.stem}.{x}.jpg')
        cv2.imwrite(str(augmented_image_path), augmented['image'])

        annotation = {'image': image_path.name, 'bbox': [0, 0, 0, 0], 'class': 0}

        if label_path.exists():
            if augmented['bboxes']:
                annotation['bbox'] = augmented['bboxes'][0]
                annotation['class'] = 1

        with open(Path('aug_data', partition, 'labels', f'{image_path.stem}.{x}.json'), 'w') as file:
            json.dump(annotation, file)


# %%
for partition in ['train', 'test', 'validation']:
    data_dir = Path('data', partition)
    images_dir = data_dir / 'images'
    labels_dir = data_dir / 'labels'

    for image_path in images_dir.iterdir():
        if image_path.is_file():
            label_path = labels_dir / f'{image_path.stem}.json'
            try:
                process_image(image_path, label_path, partition)
            except Exception as e:
                print(e)


# %% md
## Load augmented images in Tensorflow Dataset
# %%
def load_and_preprocess_image(path):
    image = tf.io.read_file(path)
    image = tf.image.decode_jpeg(image, channels=3)
    image = tf.image.resize(image, [SCALE_DOWN_WIDTH, SCALE_DOWN_HEIGHT])
    image = image / 255
    return image


# %%
def create_dataset(image_dir):
    dataset = tf.data.Dataset.list_files(image_dir + '/*.jpg', shuffle=False)
    dataset = dataset.map(load_and_preprocess_image)
    return dataset


# %%
train_images = create_dataset('aug_data/train/images')
test_images = create_dataset('aug_data/test/images')
val_images = create_dataset('aug_data/validation/images')


# %% md
## Prepare Labels
# %%
def load_labels(label_path):
    #   Read json file
    with open(label_path.numpy(), 'r', encoding='utf-8') as f:
        label = json.load(f)

    return [label['class']], label['bbox']


# %%
datasets = {}

for partition in ['train', 'test', 'validation']:
    json_files_pattern = str(Path('aug_data', partition, 'labels', '*.json'))
    dataset = tf.data.Dataset.list_files(json_files_pattern, shuffle=False)
    dataset = dataset.map(lambda x: tf.py_function(load_labels, [x], [tf.uint8, tf.float16]))
    datasets[partition] = dataset


# %% md
## Create Final Datasets (Image/Labels)
# %%
def prepare_dataset(images, labels, shuffle_size, batch_size=8, prefetch_size=4):
    dataset = tf.data.Dataset.zip((images, labels))
    dataset = dataset.shuffle(shuffle_size)  # reduce variance and ensure that models remain general and overfit less
    dataset = dataset.batch(batch_size)  # each batch is 8 img, 8 labels
    dataset = dataset.prefetch(prefetch_size)  # helps with bottleneck
    return dataset


# %%
train = prepare_dataset(train_images, datasets['train'], shuffle_size=5000)
test = prepare_dataset(test_images, datasets['test'], shuffle_size=1300)
validation = prepare_dataset(val_images, datasets['validation'], shuffle_size=1000)
# %%
data_samples = train.as_numpy_iterator()
# %%
res = data_samples.next()
# %%
fig, ax = plt.subplots(ncols=4, figsize=(20, 20))
for idx in range(4):
    sample_image = res[0][idx].copy()
    sample_coords = res[1][1][idx]

    cv2.rectangle(sample_image,
                  tuple(np.multiply(sample_coords[:2], [SCALE_DOWN_WIDTH, SCALE_DOWN_HEIGHT]).astype(int)),
                  tuple(np.multiply(sample_coords[2:], [SCALE_DOWN_WIDTH, SCALE_DOWN_HEIGHT]).astype(int)),
                  (255, 0, 0), 1)

    ax[idx].imshow(sample_image)
# %% md
## Build Deep Learning Neural Network
# %%
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, Dense, GlobalMaxPooling2D
from tensorflow.keras.applications import VGG16

# %%
vgg = VGG16(include_top=False)


# %%
def build_model():
    input_layer = Input(shape=(SCALE_DOWN_WIDTH, SCALE_DOWN_HEIGHT, 3))

    vgg = VGG16(include_top=False)(input_layer)

    f1 = GlobalMaxPooling2D()(vgg)
    class1 = Dense(2048, activation='relu')(f1)
    class2 = Dense(1, activation='sigmoid')(class1)

    f2 = GlobalMaxPooling2D()(vgg)
    regress1 = Dense(2048, activation='relu')(f2)
    regress2 = Dense(4, activation='sigmoid')(regress1)

    waldo_finder = Model(inputs=input_layer, outputs=[class2, regress2])
    return waldo_finder


# %%
waldo_finder = build_model()
# %% md
## Loss functions
# %%
# 75% of the original learning rate after each epoch
batches_per_epoch = len(train)
lr_decay = (1. / 0.75 - 1) / batches_per_epoch
# %%
opt = tf.keras.optimizers.legacy.Adam(learning_rate=0.0001, decay=lr_decay)


# %%
def localization_loss(y_true, yhat):
    delta_coord = tf.reduce_sum(tf.square(y_true[:, :2] - yhat[:, :2]))

    height_true = y_true[:, 3] - y_true[:, 1]
    width_true = y_true[:, 2] - y_true[:, 0]

    height_pred = yhat[:, 3] - yhat[:, 1]
    width_pred = yhat[:, 2] - yhat[:, 0]

    delta_size = tf.reduce_sum(tf.square(width_true - width_pred) + tf.square(height_true - height_pred))

    return delta_coord + delta_size


# %%
classloss = tf.keras.losses.BinaryFocalCrossentropy()
regressloss = localization_loss


# %% md
## Train Neural Network
# %%
class WaldoFinder(Model):
    def __init__(self, waldo_finder, **kwargs):
        super().__init__(**kwargs)
        self.model = waldo_finder

    def compile(self, opt, classloss, localizationloss, **kwargs):
        super().compile(**kwargs)
        self.closs = classloss
        self.lloss = localizationloss
        self.opt = opt

    def train_step(self, batch, **kwargs):
        X, y = batch

        with tf.GradientTape() as tape:
            classes, coords = self.model(X, training=True)

            batch_classloss = self.closs(y[0], classes)
            batch_localizationloss = self.lloss(tf.cast(y[1], tf.float32), coords)

            total_loss = batch_localizationloss + 0.5 * batch_classloss

            grad = tape.gradient(total_loss, self.model.trainable_variables)

        opt.apply_gradients(zip(grad, self.model.trainable_variables))

        return {"total_loss": total_loss, "class_loss": batch_classloss, "regress_loss": batch_localizationloss}

    def test_step(self, batch, **kwargs):
        X, y = batch

        classes, coords = self.model(X, training=False)

        batch_classloss = self.closs(y[0], classes)
        batch_localizationloss = self.lloss(tf.cast(y[1], tf.float32), coords)
        total_loss = batch_localizationloss + 0.5 * batch_classloss

        return {"total_loss": total_loss, "class_loss": batch_classloss, "regress_loss": batch_localizationloss}

    def call(self, X, **kwargs):
        return self.model(X, **kwargs)


# %%
model = WaldoFinder(waldo_finder)
# %%
model.compile(opt, classloss, regressloss)
# %% md
## Train
# %%
logdir = 'logs'
# %%
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=logdir)
# %%
hist = model.fit(train, epochs=20, validation_data=validation, callbacks=[tensorboard_callback])
# %%
data_keys = [
    ('total_loss', 'val_total_loss', 'Loss'),
    ('class_loss', 'val_class_loss', 'Classification Loss'),
    ('regress_loss', 'val_regress_loss', 'Regression Loss')
]

fig, axes = plt.subplots(ncols=3, figsize=(20, 5))

for ax, (key1, key2, title) in zip(axes, data_keys):
    ax.plot(hist.history[key1], color='teal', label=key1.replace('_', ' '))
    ax.plot(hist.history[key2], color='orange', label=key2.replace('_', ' '))
    ax.set_title(title)
    ax.legend()

plt.show()
# %%
test_data = test.as_numpy_iterator()
# %%
test_sample = test_data.next()
# %%
yhat = waldo_finder.predict(test_sample[0])
# %%
fig, ax = plt.subplots(ncols=4, figsize=(20, 20))
for idx in range(4):
    sample_image = test_sample[0][idx].copy()
    sample_coords = yhat[1][idx]

    if yhat[0][idx] > 0.5:
        cv2.rectangle(sample_image,
                      tuple(np.multiply(sample_coords[:2], [SCALE_DOWN_WIDTH, SCALE_DOWN_HEIGHT]).astype(int)),
                      tuple(np.multiply(sample_coords[2:], [SCALE_DOWN_WIDTH, SCALE_DOWN_HEIGHT]).astype(int)),
                      (255, 0, 0), 1)
    ax[idx].imshow(sample_image)
# %% md
## Save model
# %%
from tensorflow.keras.models import load_model

# %%
waldo_finder.save('waldo_finder.h5')
# %%
oh_here_is_waldo = load_model('waldo_finder.h5')