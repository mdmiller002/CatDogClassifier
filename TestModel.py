from keras.models import model_from_json
import numpy as np
from keras.preprocessing import image
import os
import time

def main():

    # Load the model from saved files
    with open('classifier.json', 'r') as json_file:
        loaded_model_json = json_file.read()

    classifier = model_from_json(loaded_model_json)
    classifier.load_weights('classifier_weights.h5')
    classifier.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    times = []
    path = 'data/test_set/'
    correctGuesses = 0
    numGuesses = 0
    for root, dirs, files in os.walk(path):
        for imgFile in files:
            if imgFile.endswith('.jpg'):

                imgPath = os.path.join(root, imgFile)
                start = time.time()
                test_image = image.load_img(imgPath, target_size=(64, 64))
                test_image = image.img_to_array(test_image)
                test_image = np.expand_dims(test_image, axis=0)
                result = classifier.predict_classes(test_image)
                end = time.time()

                if result[0][0] == 0 and 'cat' in imgPath:
                    correctGuesses += 1
                elif result[0][0] == 1 and 'dog' in imgPath:
                    correctGuesses += 1
                numGuesses += 1

                times.append((end - start) * 1000)

    accuracy = correctGuesses / numGuesses * 100
    print('Average accuracy: %3.2f%%' % accuracy)

    averageTime = sum(times) / len(times)
    print('Average time: %8.2fms' % averageTime)


if __name__ == '__main__':
    main()