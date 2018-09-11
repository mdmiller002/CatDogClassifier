"""
This module tests the model built via BuildModel
"""

from keras.models import load_model
import numpy as np
from keras.preprocessing import image
import os
import time


def main():

    classifier = load_model('model.h5')

    # Paths to the testing data sets
    catPath = 'data/test_set/cats'
    dogPath = 'data/test_set/dogs'

    # Track average times, and cat & dog accuracy
    times = []
    correctCatGuesses = 0
    numCatGuesses = 0
    correctDogGuesses = 0
    numDogGuesses = 0

    # Run inferences on the cat images
    for imgFile in os.listdir(catPath):
        if imgFile.endswith('.jpg'):
            imgPath = os.path.join(catPath, imgFile)
            start = time.time()
            test_image = image.load_img(imgPath, target_size=(150, 150))
            test_image = image.img_to_array(test_image)
            test_image = np.expand_dims(test_image, axis=0)
            result = classifier.predict_classes(test_image)
            end = time.time()

            if result[0][0] == 0:
                correctCatGuesses += 1
            numCatGuesses += 1

            times.append((end - start) * 1000)

    # Run inferences on the dog images
    for imgFile in os.listdir(dogPath):
        if imgFile.endswith('jpg'):
            imgPath = os.path.join(dogPath, imgFile)
            start = time.time()
            test_image = image.load_img(imgPath, target_size=(150, 150))
            test_image = image.img_to_array(test_image)
            test_image = np.expand_dims(test_image, axis=0)
            result = classifier.predict_classes(test_image)
            end = time.time()

            if result[0][0] == 1:
                correctDogGuesses += 1
            numDogGuesses += 1

            times.append((end - start) * 1000)

    # Calculate average accuracies and times of the runs
    accuracy = (correctCatGuesses + correctDogGuesses) / (numCatGuesses + numDogGuesses) * 100
    catAccuracy = correctCatGuesses / numCatGuesses * 100
    dogAccuracy = correctDogGuesses / numDogGuesses * 100
    averageTime = sum(times) / len(times)

    print('Average accuracy: %3.2f%%' % accuracy)
    print('Average cat accuracy: %3.2f%%' % catAccuracy)
    print('Average dog accuracy: %3.2f%%' % dogAccuracy)
    print('Average time: %8.2fms' % averageTime)


if __name__ == '__main__':
    main()