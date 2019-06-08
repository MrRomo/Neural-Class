from neural_class import NeuralClass
from PersonClassifier2 import PersonClassifier

import cv2
import time
import matplotlib.pyplot as plt
import glob
import os
import numpy as np


def camera():
    batch = []
    print("cropping")
    cap = cv2.VideoCapture(0)
    # frame = cv2.imread('../Resources/vieja2.jpg')
    for i in range(10):
        print "capture image {}".format(str(i))
        error, frame = cap.read()
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        batch.append(frame)

    cap.release()

    return batch


def files():

    img_dir = "../Resources"  # Enter Directory of all images
    data_path = os.path.join(img_dir, '*g')
    files = glob.glob(data_path)
    data = []
    for f1 in files:
        img = cv2.imread(f1)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        data.append(img)

    return data


if __name__ == "__main__":

    batch = files()

    neural = NeuralClass(batch, 0.1)
    print(neural.detect())
    print(neural.age())
    personC = PersonClassifier()
    columns = len(neural.faces)
    rows = 2
    fig, ax = plt.subplots(rows, columns)
    data = []
    for frame in neural.frame:
        res = personC.gender_race(frame)
        data.append([res["Gender"], res["Race"], res["Age"],
                        res["ColorHair"], res["Glasses"]])

    fig.suptitle(
        'Faces Detected\n {}/{}\n{}'.format(columns, len(batch), data))
    print columns
    for i in range(columns):
        for j in range(rows):
            print i, j

            ax[j][i].imshow(neural.frame[i])
            ax[j][i].set_yticklabels([])
            ax[j][i].set_xticklabels([])
        ax[j][i].set_xlabel("{}%".format(neural.percents[i]))
        ax[j][i].imshow(neural.faces[i])
        ax[j][i].set_yticklabels([])
        ax[j][i].set_xticklabels([])

 
    plt.show()
