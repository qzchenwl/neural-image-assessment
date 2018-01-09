from keras.models import Model
from keras.layers import Dense, Dropout
from keras.applications.mobilenet import MobileNet
from keras.callbacks import ModelCheckpoint, TensorBoard
from keras.optimizers import Adam
from utils import *
from losses import emd

def main():
    base_model = MobileNet((224,224,3), alpha=1, include_top=False, pooling='avg')
    for layer in base_model.layers:
        layer.trainable = False

    x = Dropout(0.75)(base_model.output)
    x = Dense(10, activation='softmax')(x)

    model = Model(base_model.input, x)
    model.summary()

    optimizer = Adam(lr=1e-3)
    model.compile(optimizer, loss=emd)

    checkpoint = ModelCheckpoint('weights/mobilenet_weights.h5', monitor='val_loss', verbose=1, save_weights_only=True, save_best_only=True, mode='min')
    tensorboard = TensorBoard()
    callbacks = [checkpoint, tensorboard]

    batch_size = 5
    epochs = 5

    filenames, labels = list_images('/data/raid10/test/AVA_dataset/AVA_test.txt')
    train_filenames, train_labels, test_filenames, test_labels = train_test_split(filenames, labels, train_size=0.8)
    train_filenames, train_labels, val_filenames, val_labels = train_test_split(train_filenames, train_labels, train_size = 0.8)

    train_generator = generate(train_filenames, train_labels, batch_size, shuffle_size=100, processing_fn=training_preprocess_mobilenet)
    val_generator = generate(val_filenames, val_labels, batch_size, shuffle_size=100, processing_fn=val_preprocess_mobilenet)

    model.fit_generator(train_generator,
                        steps_per_epoch=(53//batch_size),
                        epochs=epochs,
                        verbose=1,
                        callbacks=callbacks,
                        validation_data=val_generator,
                        validation_steps=(14//batch_size)
                        )

main()
