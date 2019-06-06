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
        time.sleep(0.5)
        batch.append(frame)

    cap.release()
    neural = NeuralClass(batch, 0.1)

    columns = 2 
    rows = len(neural.faces)
    fig, ax = plt.subplots(rows, columns)
    fig.suptitle('Faces Detected\n {}/{}'.format(rows, len(batch)))

    for i in range(rows):
        for j in range(columns):
            ax[i][j].imshow(neural.faces[i])
            ax[i][j].set_yticklabels([])
            ax[i][j].set_xticklabels([])
        ax[i][j].set_xlabel("{}%".format(neural.percents[i]))
        ax[i][0].imshow(neural.frame[i])
        ax[i][j].set_yticklabels([])
        ax[i][j].set_xticklabels([])

    plt.show()
