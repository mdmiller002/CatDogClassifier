"""
This module builds a model that is a binary classifier
"""

from keras.models import Sequential
from keras import layers
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import TensorBoard, ModelCheckpoint
import numpy as np
from time import time


def BuildModel():
    """
    Build the classifier model
    :return: Sequential model instance of the built network
    """

    batchSize = 16
    epochs = 5

    # Create the keras model
    classifier = Sequential()

    classifier.add(layers.Conv2D(32, (3, 3), input_shape=(150, 150, 3)))
    classifier.add(layers.Activation('relu'))
    classifier.add(layers.MaxPooling2D(pool_size=(2, 2)))

    classifier.add(layers.Conv2D(32, (3, 3)))
    classifier.add(layers.Activation('relu'))
    classifier.add(layers.MaxPooling2D(pool_size=(2, 2)))

    classifier.add(layers.Conv2D(64, (3, 3)))
    classifier.add(layers.Activation('relu'))
    classifier.add(layers.MaxPooling2D(pool_size=(2, 2)))

    classifier.add(layers.Flatten())

    classifier.add(layers.Dense(units=128))
    classifier.add(layers.Activation('relu'))
    classifier.add(layers.Dropout(0.5))
    classifier.add(layers.Dense(1))
    classifier.add(layers.Activation('sigmoid'))

    classifier.compile(loss='binary_crossentropy',
                  optimizer='rmsprop',
                  metrics=['accuracy'])

    # Augment training and tesing images
    train_datagen = ImageDataGenerator(rescale=1./255,
                                       shear_range=0.2,
                                       zoom_range=0.2,
                                       horizontal_flip=True)

    test_datagen = ImageDataGenerator(rescale=1./255)

    # Create generators for training and testing data
    trainingGenerator = train_datagen.flow_from_directory('data/training_set',
                                                     target_size=(150, 150),
                                                     batch_size=batchSize,
                                                     class_mode='binary')


    testingGenerator = test_datagen.flow_from_directory('data/test_set',
                                                target_size=(150, 150),
                                                batch_size=batchSize,
                                                class_mode='binary')

    with open('classes.txt', 'w') as classesFile:
        classesFile.write(str(trainingGenerator.class_indices))

    # Callbacks
    tensorboardCb = TensorBoard(log_dir="logs/{}".format(time()))
    checkpointCb = ModelCheckpoint('model_checkpoint.h5', monitor='val_acc', verbose=1, save_best_only=True, mode='max')

    callbacksList = [checkpointCb]

    classifier.fit_generator(trainingGenerator,
                             steps_per_epoch= 2000 // batchSize,
                             epochs=epochs,
                             validation_data=testingGenerator,
                             validation_steps=800 // batchSize,
                             callbacks=callbacksList)

    classifier.save('model.h5')

    return classifier

if __name__ == '__main__':
    BuildModel()
