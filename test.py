from neural_class import NeuralClass
import cv2
import time
import matplotlib.pyplot as plt

if __name__ == "__main__":

    batch = list()
    print("cropping")
    cap = cv2.VideoCapture(0)
    # frame = cv2.imread('../Resources/vieja2.jpg')
    for i in range(10):
        print "capture image {}".format(str(i))
        error, frame = cap.read()
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        batch.append(frame)

    cap.release()
    neural = NeuralClass(batch, 0.1)

    columns = 2
    rows = len(neural.faces)
    fig, ax = plt.subplots(rows, columns)

    for i in range(rows):
        for j in range(columns):
            ax[i][j].imshow(neural.faces[i])
        ax[i][0].imshow(neural.frame[i])

    print(ax)
    ax.tranpose()
    print(ax)
    print(type(ax))
    plt.show()
