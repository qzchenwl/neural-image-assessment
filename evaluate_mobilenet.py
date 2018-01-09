import numpy as np
from keras.models import Model
from keras.layers import Dense, Dropout
from keras.applications.mobilenet import MobileNet

from utils import parse_fn, val_preprocess_mobilenet

def add_new_top_layer(base_model):
    """
    Add top layer to the net
    Args:
        base_model: keras model excluding top
    """
    x = base_model.output
    x = Dropout(0.75)(x)
    x = Dense(10, activation='softmax')(x)
    model = Model(input=base_model.input, output=x)
    return model

# setup model
base_model = MobileNet(input_shape=(224,224,3), alpha=1, include_top=False, pooling='avg')
model = add_new_top_layer(base_model)
model.load_weights('weights/mobilenet_weights.h5')

def predict(image_paths):
    X = np.array(list(
        map(lambda path: val_preprocess_mobilenet(parse_fn(path)), image_paths)
    ))
    scores = model.predict(X, batch_size=32, verbose=1)
    return scores

