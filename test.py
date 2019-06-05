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
        batch.append(frame)

    cap.release()
    neural = NeuralClass(batch)
    print ("befores")
    print (neural.coord)
    print type(neural.coord)
    print len(neural.coord)
    fig = plt.figure(figsize=(8, 8))
    columns = 2
    rows = 5
    for i in range(0, columns*rows):
        print("plot {}".format(i))
        if(i > len(neural.faces)-1):
            img = neural.faces[len(neural.faces)-1]
        else:
            img = neural.faces[i]
        fig.add_subplot(rows, columns, i+1)

        plt.imshow(img)
    for i in range(0, columns*rows):
        print("plot {}".format(i))
        img = neural.frame[i]
        fig.add_subplot(rows, columns, i+1)
        plt.imshow(img)
    plt.show()
