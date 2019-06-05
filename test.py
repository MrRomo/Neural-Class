from neural_class import NeuralClass
import cv2
import time

if __name__ == "__main__":

    batch = list()
    print("cropping")
    cap = cv2.VideoCapture(0)
    # frame = cv2.imread('../Resources/vieja2.jpg')
    for i in range(10):
        time.sleep(0.1)
        print "capture image {}".format(str(i))
        error, frame = cap.read()
        batch.append(frame)

    cap.release()
    neural = NeuralClass(batch)

    print(len(neural.faces))
    print(len(neural.frame))
    for face in neural.faces:
        print (face["cord"])
