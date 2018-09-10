from keras.models import Sequential
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import TensorBoard
from keras.models import  model_from_json
import numpy as np
from time import time
import csv


def BuildModel():
    classifier = Sequential()
    classifier.add(Conv2D(32, (3, 3), input_shape=(64, 64, 3), activation='relu'))
    classifier.add(MaxPooling2D(pool_size=(2, 2)))
    classifier.add(Flatten())
    classifier.add(Dense(units=128, activation='relu'))
    classifier.add(Dense(units=1, activation='sigmoid'))
    classifier.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    train_datagen = ImageDataGenerator(rescale=1. / 255,
                                       shear_range=0.2,
                                       zoom_range=0.2,
                                       horizontal_flip=True)

    test_datagen = ImageDataGenerator(rescale=1. / 255)

    training_set = train_datagen.flow_from_directory('data/training_set',
                                                     target_size=(64, 64),
                                                     batch_size=32,
                                                     class_mode='binary')

    with open('classes.txt', 'w') as classesFile:
        classesFile.write(str(training_set.class_indices))


    test_set = test_datagen.flow_from_directory('data/test_set',
                                                target_size=(64, 64),
                                                batch_size=32,
                                                class_mode='binary')

    tensorboard = TensorBoard(log_dir="logs/{}".format(time()))
    classifier.fit_generator(training_set,
                             steps_per_epoch=8000,
                             epochs=5,
                             validation_data=test_set,
                             validation_steps=500,
                             callbacks=[tensorboard])

    model_json = classifier.to_json()
    with open("classifier.json", "w") as json_file:
        json_file.write(model_json)
    classifier.save_weights('classifier_weights.h5')

if __name__ == '__main__':
    BuildModel()
