import numpy as np

from keras.models import Model
from keras.layers import Dense, Dropout
from keras.applications.inception_resnet_v2 import InceptionResNetV2
from keras.preprocessing.image import load_img, img_to_array
#from image_preprocessing import centerCrop224
from keras.applications.imagenet_utils import preprocess_input
from PIL import ImageOps, Image
import tensorflow as tf
from path import Path

IMAGE_SIZE = 224

def resize(img):
    desired_size = IMAGE_SIZE
    img.thumbnail((desired_size,desired_size), Image.LANCZOS)
    delta_w = desired_size - img.width
    delta_h = desired_size - img.height
    padding = (delta_w//2, delta_h//2, delta_w-(delta_w//2), delta_h-(delta_h//2))
    new_im = ImageOps.expand(img, padding)
    return new_im


base_model = InceptionResNetV2(input_shape=(None, None, 3),
                               include_top=False, pooling='avg', weights=None)


x = Dropout(0.75)(base_model.output)
x = Dense(10, activation='softmax', name='toplayer')(x)

model = Model(base_model.input, x)
model.load_weights('weights/inceptresnet_weights4.h5')

def predict(image_paths):
    X = np.array(list(
        map(lambda path: preprocess_input(img_to_array(resize(load_img(path))), mode='tf'), image_paths)
    ))
    scores = model.predict(X, batch_size=32, verbose=1)
    return scores


