import PIL
import numpy as np
from keras.preprocessing.image import load_img, img_to_array, array_to_img, flip_axis

VGG_MEAN = [123.68, 116.78, 103.94]

def list_images(file):
    """
    Get all the images and labels in file
    """
    with open(file, 'r') as f:
        lines = f.readlines()

def random_crop(img, size):
    """
    Args:
        img: Input image to crop.
        size: 2-D tuple, (height, width)
    """

    height, width, channels = img.shape
    crop_h, crop_w = size

    range_h = height - crop_h
    range_w = width -crop_w

    offset_h = 0 if range_h == 0 else np.random.randint(range_h)
    offset_w = 0 if range_w == 0 else np.random.randint(range_w)

    crop_img = img[offset_h : offset_h + crop_h, offset_w : offset_w + crop_w, : ]

    return crop_img

def center_crop(img, size):
    """
    Args:
        img: Input image to crop.
        size: 2-D tuple, (height, width)
    """

    height, width, channels = img.shape
    crop_h, crop_w = size

    center_h, center_w = height // 2, width // 2
    half_h, half_w = crop_h // 2, crop_w // 2

    crop_img = img[
        center_h - half_h : center_h + half_h,
        center_w - half_w : center_w + half_w,
        :]

    return crop_img

def random_flip_left_right(img):
    if np.random.random() < 0.5:
        return flip_axis(img, axis=1)
    return img


# Preprocessing (for both training and validation):
# (1) Decode the image from jpg format
# (2) Resize the image so its smaller side is 256 pixels long
def parse_function(filename, label):
    img = load_img(filename)

    smallest_side = 256.0
    scale = smallest_side / img.width if img.height > img.width else smallest_side / img.height

    resized_img = img.resize((int(img.width * scale), int(img.height * scale)), PIL.Image.BILINEAR)
    img_array = img_to_array(resized_img)

    return img_array, label

# Preprocessing (for training)
# (3) Take a random 224x224 crop to the scaled image
# (4) Horizontally flip the image with probability 1/2
# (5) Substract the per color mean `VGG_MEAN`
# Note: we don't normalize the data here, as VGG was trained without normalization
def training_preprocess_vgg(img, label):
    crop_img = random_crop(img, (224, 224))
    flip_img = random_flip_left_right(crop_img)

    means = np.reshape(VGG_MEAN, [1, 1, 3])
    centered_img = flip_img - means

    return centered_img, label

def training_preprocess_mobilenet(img, label):
    crop_img = random_crop(img, (224, 224))
    flip_img = random_flip_left_right(crop_img)

    normalized_img = (flip_img - 127.5) / 127.5

    return normalized_img, label


# Preprocessing (for validation)
# (3) Take a central 224x224 crop to the scaled image
# (4) Substract the per color mean `VGG_MEAN`
# Note: we don't normalize the data here, as VGG was trained without normalization
def val_preprocess_vgg(img, label):
    crop_img = center_crop(img, (224, 224))

    means = np.reshape(VGG_MEAN, [1, 1, 3])
    centered_img = crop_img - means

    return centered_img

def val_preprocess_mobilenet(img, label):
    crop_img = center_crop(img, (224, 224))

    normalized_img = (flip_img - 127.5) / 127.5

    return normalized_img, label


