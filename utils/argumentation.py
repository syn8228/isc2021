import math
import numbers
import random
import numpy as np
from torchvision import transforms

from PIL import Image, ImageOps, ImageFilter


class VerticalFlip(object):
    def __init__(self, probability=0.5):
        self.p = probability
        
    def __call__(self, image):
        image = np.array(image)
        if random.random() < self.p:
            image = np.flip(image, 1).copy()
        image = Image.fromarray(np.uint8(image))
        return image


class HorizontalFlip(object):
    def __init__(self, probability=0.5):
        self.p = probability

    def __call__(self, image):
        image = np.array(image)
        if random.random() < self.p:
            image = np.flip(image, 0).copy()
        image = Image.fromarray(np.uint8(image))
        return image


class GaussianBlur(object):
    def __init__(self, probability=0.5):
        self.p = probability

    def __call__(self, image):
        if random.random() < self.p:
            radius = random.uniform(0.25, 3)
            image = image.filter(ImageFilter.GaussianBlur(radius=radius))
        return image


class Rotate(object):
    def __init__(self, probability=0.5):
        self.p = probability

    def __call__(self, image):
        if random.random() < self.p:
            angle = random.randint(0, 360)
            image = image.rotate(angle)
        return image


class GaussianNoise(object):
    def __init__(self, probability=0.5):
        self.p = probability

    def __call__(self, image):
        mean = 0
        std = 50
        if random.random() < self.p:
            image = np.asarray(image, dtype = np.float32)
            noise = np.random.normal(mean, std, size=image.shape)
            image = image + noise
            image[image > 255] = 255
            image[image < 0] = 0
            image = np.asarray(image, dtype=np.uint8)
            image = Image.fromarray(np.uint8(image))
        return image


class ColRec(object):
    def __init__(self, probability=0.5, n_rectangle=10):
        self.p = probability
        self.n_rect = n_rectangle

    def __call__(self, image):
        x_size = 15
        y_size = 15
        if random.random() < self.p:

            image = np.array(image)
            im_y, im_x = image.shape[0], image.shape[1]
            for i in range(self.n_rect):
                x = np.random.randint(im_x - x_size)
                y = np.random.randint(im_y - y_size)
                color = np.random.randint(255, size=(1, 1, 3))
                image[y: y + y_size, x: x + x_size] = color
            image = Image.fromarray(np.uint8(image))
        return image


class ZoomIn(object):
    def __init__(self, probability=0.5):
        self.p = probability

    def __call__(self, image):
        box_max = 0.8
        box_min = 0.5
        if random.random() < self.p:
            image = np.array(image)
            im_y, im_x = image.shape[0], image.shape[1]
            size_y = np.random.randint(im_y * box_min, im_y * box_max)
            size_x = np.random.randint(im_x * box_min, im_x * box_max)
            x = im_x - size_x
            y = im_y - size_y
            image = image[y: y+size_y, x: x+size_x]
            image = Image.fromarray(np.uint8(image))
            image = image.resize((im_y, im_x))
        return image


class ZoomOut(object):
    def __init__(self, probability=0.5):
        self.p = probability

    def __call__(self, image):
        zoom_max = 1.5
        zoom_min = 1.3
        if random.random() < self.p:
            image = np.array(image)
            im_y, im_x = image.shape[0], image.shape[1]
            zoom_factor = np.random.uniform(zoom_min, zoom_max)
            size_y = int(im_y * zoom_factor)
            size_x = int(im_x * zoom_factor)
            image_black = np.zeros((size_y, size_x, 3), dtype=np.uint8)
            x = np.random.randint(size_x - im_x)
            y = np.random.randint(size_y - im_y)
            image_black[y: y + im_y, x: x + im_x] = image
            image = np.asarray(image_black, dtype=np.uint8)
            image = Image.fromarray(np.uint8(image))
            image = image.resize((im_y, im_x))
        return image


class RandomCut(object):
    def __init__(self, probability=0.5):
        self.p = probability

    def __call__(self, image):
        if random.random() < self.p:
            transform = transforms.RandomResizedCrop(image.size)
            image = transform(image)
        return image


class NegativeImage(object):
    def __init__(self, probability=0.5):
        self.p = probability

    def __call__(self, image):
        if random.random() < self.p:
            image = np.array(image)
            image = np.ones(image.shape)*255 - image
            image = Image.fromarray(np.uint8(image))
        return image


class MergeImage(object):
    def __init__(self, background, probability=0.5):
        self.p = probability
        self.bg_image = background

    def __call__(self, image):
        zoom_max = 1.5
        zoom_min = 1.3
        if random.random() < self.p:
            image = np.array(image)
            im_y, im_x = image.shape[0], image.shape[1]
            zoom_factor = np.random.uniform(zoom_min, zoom_max)
            size_y = int(im_y * zoom_factor)
            size_x = int(im_x * zoom_factor)
            bg_img = self.bg_image.resize((size_x, size_y))
            bg_img = np.array(bg_img)
            x = np.random.randint(size_x - im_x)
            y = np.random.randint(size_y - im_y)
            bg_img[y: y + im_y, x: x + im_x] = image
            image = np.asarray(bg_img, dtype=np.uint8)
            image = Image.fromarray(np.uint8(image))
            image = image.resize((im_y, im_x))
        return image

# background = np.ones((50, 53, 3))
# img = np.ones((100, 203, 3))
# img_pil = Image.fromarray(np.uint8(img))
# background_pil = Image.fromarray(np.uint8(background))
# list = [
#     NegativeImage(1.0),
#     RandomCut(1.0),
#     MergeImage(background_pil, 1.0),
#     ZoomOut(1.0),
#     HorizontalFlip(1.0),
# ]
# random.shuffle(list)
# trans = transforms.Compose(list)
# img = trans(img_pil)
# print(img.size)
