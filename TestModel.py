"""
This module tests the model built via BuildModel
"""

from keras.models import load_model
import numpy as np
from keras.preprocessing import image
import os
import sys
import time



def main():

    modelFile = 'model.h5'
    testDirectory = 'data/test_set/'
    CAT_CLASS = 0
    DOG_CLASS = 1

    classifier = load_model(modelFile)


    # Track average times, and cat & dog accuracy
    times = []
    correctCatGuesses = 0
    numCatGuesses = 0
    correctDogGuesses = 0
    numDogGuesses = 0

    # Run inferences on all images in the testing directory
    for root, dirs, files in os.walk(testDirectory):
        for imgFile in files:
            if imgFile.endswith('.jpg'):

                # Run the inference on the image
                imgPath = os.path.join(root, imgFile)
                start = time.time()
                test_image = image.load_img(imgPath, target_size=(150, 150))
                test_image = image.img_to_array(test_image)
                test_image = np.expand_dims(test_image, axis=0)
                result = classifier.predict_classes(test_image)
                end = time.time()

                if 'cats' in imgPath:
                    if result[0][0] == CAT_CLASS:
                        correctCatGuesses += 1
                    numCatGuesses += 1

                if 'dogs' in imgPath:
                    if result[0][0] == DOG_CLASS:
                        correctDogGuesses += 1
                    numDogGuesses += 1

                times.append((end - start) * 1000)
                print('*', end='')
                sys.stdout.flush()
    print()


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