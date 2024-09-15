import torch
import torchvision
from torchvision.transforms import v2


def augment_image(images):
    augmented_images = []
    for image in images:

