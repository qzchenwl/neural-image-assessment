from keras.models import Model
from keras.layers import Dense, Dropout
from keras.applications.vgg16 import VGG16
from keras.callbacks import ModelCheckpoint, TensorBoard
from keras.optimizers import Adam, SGD
from utils import list_images, training_preprocess_vgg, val_preprocess_vgg, generate
from losses import emd
from data import TRAIN_DATASET, VAL_DATASET

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

def setup_transfer_learning(model, base_model):
    for layer in base_model.layers:
        layer.trainable = False
    model.compile(optimizer=Adam(lr=1e-3), loss=emd)

def setup_finetuning(model):
    for layer in model.layers:
        layer.trainable = True
    model.compile(optimizer=SGD(lr=1e-4, momentum=0.9), loss=emd)

def train(args):
    # prepare data
    train_filenames, train_labels = list_images(TRAIN_DATASET)
    val_filenames, val_labels = list_images(VAL_DATASET)

    train_generator = generate(train_filenames, train_labels, batch_size=args.batch_size, shuffle_size=args.shuffle_size, processing_fn=training_preprocess_vgg)
    val_generator   = generate(val_filenames,   val_labels,   batch_size=args.batch_size, shuffle_size=args.shuffle_size, processing_fn=val_preprocess_vgg)

    train_sample_size = len(train_filenames)
    val_sample_size   = len(val_filenames)

    checkpoint = ModelCheckpoint('weights/vgg16_weights.h5', monitor='val_loss', verbose=1, save_weights_only=True, save_best_only=True, mode='min')
    tensorboard = TensorBoard()
    callbacks = [checkpoint, tensorboard]


    # setup model
    base_model = VGG16(input_shape=(224,224,3), include_top=False, pooling='avg')
    model = add_new_top_layer(base_model)

    # transfer learning
    setup_transfer_learning(model, base_model)

    model.summary()
    history_tl = model.fit_generator(
        train_generator,
        validation_data=val_generator,
        steps_per_epoch=(train_sample_size//args.batch_size),
        validation_steps=(val_sample_size//args.batch_size),
        epochs=args.epochs,
        verbose=1,
        callbacks=callbacks
    )

    # finetuning
    setup_finetuning(model)

    model.summary()
    history_ft = model.fit_generator(
        train_generator,
        validation_data=val_generator,
        steps_per_epoch=(train_sample_size//args.batch_size),
        validation_steps=(val_sample_size//args.batch_size),
        epochs=args.epochs,
        verbose=1,
        callbacks=callbacks
    )

    model.save('weights/mobilenet-ft.model')


if __name__ == '__main__':
    class atdict(dict):
        __getattr__= dict.__getitem__
        __setattr__= dict.__setitem__
        __delattr__= dict.__delitem__
    args = atdict({})
    args.batch_size = 32
    args.shuffle_size = 100
    args.epochs = 3
    train(args)
