import PIL
import random
import numpy as np
from keras.preprocessing.image import load_img, img_to_array, array_to_img, flip_axis

VGG_MEAN = [123.68, 116.78, 103.94]
AVA_IMAGE_DIR = '/data/raid10/test/AVA_dataset/images/'

def list_images(file):
    """
    Get all the images and labels in file
    """
    filenames = []
    labels = []
    with open(file, 'r') as f:
        lines = f.readlines()
        for i, line in enumerate(lines):
            tokens = line.split()
            filename = AVA_IMAGE_DIR + tokens[1] + '.jpg'
            scores = np.array(tokens[2:12], dtype='float32')
            scores /= scores.sum()
            filenames.append(filename)
            labels.append(scores)
    return filenames, labels

def train_test_split(X, y, train_size=0.9):
    """
    X_train, y_train, X_test, y_test = train_test_split(X, y, test_size)
    """
    Xy = list(zip(X, y))
    random.shuffle(Xy)
    train_count = int(len(Xy) * train_size)
    X_train, y_train = zip(*Xy[0:train_count])
    X_test, y_test = zip(*Xy[train_count:])
    return X_train, y_train, X_test, y_test

def random_crop(img, size):
    """
    Args:
        img: Input image to crop.
        size: 2-D tuple, (height, width)
    """

    height, width, channels = img.shape
    crop_h, crop_w = size

    range_h = height - crop_h
    range_w = width - crop_w

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
def parse_fn(filename):
    img = load_img(filename)

    smallest_side = 256.0
    scale = smallest_side / img.width if img.height > img.width else smallest_side / img.height

    resized_img = img.resize((int(img.width * scale), int(img.height * scale)), PIL.Image.BILINEAR)
    img_array = img_to_array(resized_img)

    return img_array

# Preprocessing (for training)
# (3) Take a random 224x224 crop to the scaled image
# (4) Horizontally flip the image with probability 1/2
# (5) Substract the per color mean `VGG_MEAN`
# Note: we don't normalize the data here, as VGG was trained without normalization
def training_preprocess_vgg(img):
    crop_img = random_crop(img, (224, 224))
    flip_img = random_flip_left_right(crop_img)

    means = np.reshape(VGG_MEAN, [1, 1, 3])
    centered_img = flip_img - means

    return centered_img

def training_preprocess_mobilenet(img):
    crop_img = random_crop(img, (224, 224))
    flip_img = random_flip_left_right(crop_img)

    normalized_img = (flip_img - 127.5) / 127.5

    return normalized_img


# Preprocessing (for validation)
# (3) Take a central 224x224 crop to the scaled image
# (4) Substract the per color mean `VGG_MEAN`
# Note: we don't normalize the data here, as VGG was trained without normalization
def val_preprocess_vgg(img):
    crop_img = center_crop(img, (224, 224))

    means = np.reshape(VGG_MEAN, [1, 1, 3])
    centered_img = crop_img - means

    return centered_img

def val_preprocess_mobilenet(img):
    crop_img = center_crop(img, (224, 224))

    normalized_img = (crop_img - 127.5) / 127.5

    return normalized_img

def generate(filenames, labels, batch_size, shuffle_size, processing_fn):
    filenames_and_labels = list(zip(filenames, labels))

    while True:
        # shuffle
        head = filenames_and_labels[0:shuffle_size]
        tail = filenames_and_labels[shuffle_size:]
        random.shuffle(head)
        filenames_and_labels = head + tail

        # batch
        batch = filenames_and_labels[0:batch_size]
        rest = filenames_and_labels[batch_size:]
        filenames_and_labels = rest + batch

        filenames, labels = zip(*batch)
        filenames = list(filenames)
        labels = list(labels)

        imgs = list(map(lambda img: processing_fn(img), map(lambda filename: parse_fn(filename), filenames)))

        yield imgs, labels


