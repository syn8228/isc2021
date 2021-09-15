import sys
import math
import numbers
import random
import numpy as np
from torchvision import transforms
import augly.image.transforms as imaugs
import augly.utils as utils

sys.path.append('/cluster/yinan/isc2021')

from PIL import Image, ImageOps, ImageFilter
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True


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


class ChangeColor(object):
    def __init__(self, probability=0.5):
        self.p = probability

    def __call__(self, image):
        rgb = random.randint(1, 3)
        if random.random() < self.p:
            image = np.asarray(image, dtype=np.float32)
            if rgb == 1:
                image[:, :, 0] += 50
            elif rgb == 2:
                image[:, :, 1] += 50
            else:
                image[:, :, 2] += 50
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
        if random.random() < self.p:

            image = np.array(image)
            im_y, im_x = image.shape[0], image.shape[1]
            for i in range(self.n_rect):
                x_size = random.randint(int(im_x / 10), int(im_x / 6))
                y_size = random.randint(int(im_y / 10), int(im_y / 6))
                x = np.random.randint(im_x - x_size)
                y = np.random.randint(im_y - y_size)
                color = np.random.randint(255, size=(1, 1, image.shape[-1]))
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
            image = image.resize((im_x, im_y))
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
            image = image.resize((im_x, im_y))
        return image


class RandomCut(object):
    def __init__(self, probability=0.5):
        self.p = probability

    def __call__(self, image):
        if random.random() < self.p:
            transform = transforms.RandomResizedCrop((image.size[1], image.size[0]))
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
        zoom_max = 3.5
        zoom_min = 1.5
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
            image = image.resize((im_x, im_y))
        return image


class ToGray(object):
    def __init__(self, probability=0.5):
        self.p = probability

    def __call__(self, image):
        if random.random() < self.p:
            image = np.asarray(image, dtype=np.float32)
            r, g, b = image[:, :, 0], image[:, :, 1], image[:, :, 2]
            gray = 0.2126*r + 0.7152*g + 0.0722*b
            image[:, :, 0], image[:, :, 1], image[:, :, 2] = gray, gray, gray
            image = Image.fromarray(np.uint8(image))
        return image


class AspectRatio(object):
    def __init__(self, probability=0.5):
        self.p = probability

    def __call__(self, image):
        factor = random.uniform(0, 2)
        meta = []
        aug = imaugs.ChangeAspectRatio(ratio=factor, p=self.p)
        image = aug(image, metadata=meta)
        return image


class Colorjitter(object):
    def __init__(self, probability=0.5):
        self.p = probability

    def __call__(self, image):
        b = random.uniform(0.5, 3)
        c = random.uniform(0.5, 3)
        s = random.uniform(0.5, 3)
        meta = []
        aug = imaugs.ColorJitter(brightness_factor=b, contrast_factor=c, saturation_factor=s, p=self.p)
        image = aug(image, metadata=meta)
        return image


class EncodingQuality(object):
    def __init__(self, probability=0.5):
        self.p = probability

    def __call__(self, image):
        q = random.randint(10, 50)
        meta = []
        aug = imaugs.EncodingQuality(quality=30, p=self.p)
        image = aug(image, metadata=meta)
        return image


class MemeFormat(object):
    def __init__(self, probability=0.5):
        self.p = probability

    def __call__(self, image):
        seed = "1234567890abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ!@#$%^&*()_+=-"
        sa = []
        len = random.randint(3, 5)
        for i in range(len):
            sa.append(random.choice(seed))
        salt = ''.join(sa)
        fg = tuple(np.random.randint(255, size=(3,)))
        bg = tuple(np.random.randint(255, size=(3,)))
        meta = []
        aug = imaugs.MemeFormat(text=salt, text_color=fg, meme_bg_color=bg, p=self.p)
        image = aug(image, metadata=meta)
        return image


class Opacity(object):
    def __init__(self, probability=0.5):
        self.p = probability

    def __call__(self, image):
        lv = random.uniform(0.5, 0.9)
        meta = []
        aug = imaugs.Opacity(level=lv, p=self.p)
        image = aug(image, metadata=meta)
        return image


class OverlayEmoji(object):
    def __init__(self, probability=0.5):
        self.p = probability

    def __call__(self, image):
        opacity = random.uniform(0.5, 1.0)
        size = random.uniform(0.2, 0.8)
        x = random.uniform(0, 1)
        y = random.uniform(0, 1)
        meta = []
        aug = imaugs.OverlayEmoji(opacity=opacity, emoji_size=size, x_pos=x, y_pos=y, p=self.p)
        image = aug(image, metadata=meta)
        covert = imaugs.ConvertColor(mode='RGB')
        image = covert(image)
        return image


class OverlayOntoScreenshot(object):
    def __init__(self, probability=0.5, background=None):
        self.p = probability
        self.bg = background

    def __call__(self, image):
        meta = []
        aug = imaugs.OverlayOntoScreenshot(p=self.p)
        image = aug(image, metadata=meta)
        return image


class OverlayText(object):
    def __init__(self, probability=0.5):
        self.p = probability

    def __call__(self, image):
        opacity = random.uniform(0.5, 1.0)
        size = random.uniform(0.1, 0.5)
        color = tuple(np.random.randint(255, size=(3,)))
        x = random.uniform(0, 0.5)
        y = random.uniform(0, 0.5)
        meta = []
        aug = imaugs.OverlayText(font_size=size, opacity=opacity, color=color, x_pos=x, y_pos=y, p=self.p)
        image = aug(image, metadata=meta)
        covert = imaugs.ConvertColor(mode='RGB')
        image = covert(image)
        return image


class AuglyRotate(object):
    def __init__(self, probability=0.5, background=None):
        self.p = probability
        self.bg = background

    def __call__(self, image):
        deg = random.uniform(0, 360)
        meta = []
        aug = imaugs.Rotate(degrees=deg, p=self.p)
        image = aug(image, metadata=meta)
        return image




# background = np.ones((50, 53, 3))
# img = np.ones((100, 203, 3))
# img_pil = Image.fromarray(np.uint8(img))
# background_pil = Image.fromarray(np.uint8(background))
# list = [
#     BrightnessImage(1.0),
# ]
# random.shuffle(list)
# trans = transforms.Compose(list)
# img = trans(img_pil)
# print(img.size)
