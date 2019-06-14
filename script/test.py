import os
import cv2
import glob
import time
import matplotlib.pyplot as plt
import numpy as np
from neural_C import NeuralClass


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


    img_dir = "../Resources/Glass"  # Enter Directory of all images
    data_path = os.path.join(img_dir, '*g')
    files = glob.glob(data_path)
    data = []
    for f1 in files:
        img = cv2.imread(f1)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        data.append(img)

    return data


if __name__ == "__main__":

    batch = camera()

    print batch


    neural = NeuralClass(batch, 0.1)

    neural.faces

    print(neural.detect())
    columns = len(neural.faces)
    rows = 2
    fig, ax = plt.subplots(rows, columns)
    res = neural.prediction
    print()
    print(neural.getAge())
    print(neural.getRace())
    print(neural.getHair())
    print(neural.getGlass())
    pred = [neural.getGender()["res"],neural.getRace()["res"],neural.getHair()["res"],neural.getAge()["res"],neural.getGlass()['res']] 
    # fig.suptitle('Faces Detected\n {}/{}\n{}'.format(columns, len(batch), pred))
    plt.figtext(.5,.9,'Faces Detected\n {}/{}\n{}'.format(columns, len(batch), pred), fontsize=20, ha='center')
    print("Columns {}".format(columns))

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
