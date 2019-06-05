import face_recognition
import cv2
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter


class NeuralClass:

    def __init__(self, frame):
        self.frame  = frame
        self.location = self.detect()
        self.age_net, self.gender_net = self.load_caffe_models()
        self.MODEL_MEAN_VALUES = (78.4263377603, 87.7689143744, 114.895847746)
        self.age_list = ['(0, 2)', '(4, 6)', '(8, 12)', '(15, 20)','(25, 32)', '(38, 43)', '(48, 53)', '(60, 100)']
        self.gender_list = ['Male', 'Female']
        self.frame = self.cropper(frame)

    def load_caffe_models(self):
        source = "../Resources/Models/"
        age_net = cv2.dnn.readNetFromCaffe(
            'deploy_age.prototxt', 'age_net.caffemodel')
        gender_net = cv2.dnn.readNetFromCaffe(
            'deploy_gender.prototxt', 'gender_net.caffemodel')
        return(age_net, gender_net)

    def genderPredictor(self):

        predict = list()
        for frame in self.frame:
            blob = cv2.dnn.blobFromImage(
                frame, 1, (227, 227), self.MODEL_MEAN_VALUES, swapRB=False)

            # Predict Gender
            self.gender_net.setInput(blob)
            gender_preds = self.gender_net.forward()
            gender = self.gender_list[gender_preds[0].argmax()]
            predict.append(gender)
        result = dict()
        gender_counter = Counter(predict)

        if(len(predict)):
            result = {
                "gender": gender_counter.most_common()[0][0],
                "percent": gender_counter.most_common()[0][1]/float(len(predict))
            }

        return result

    def agePredictor(self):
        predict = list()
        for frame in self.frame:
            blob = cv2.dnn.blobFromImage(
                frame, 1, (227, 227), self.MODEL_MEAN_VALUES, swapRB=False)
            # Predict Age
            self.age_net.setInput(blob)
            age_preds = self.age_net.forward()
            age = self.age_list[age_preds[0].argmax()]
            predict.append(age)
        age_counter = Counter(predict)
        result = dict()
        if(len(predict)):
            result = {
                "age": age_counter.most_common()[0][0],
                "percent": age_counter.most_common()[0][1]/float(len(predict))
            }
        return result

    def cropper(self, batch):
        faces = list()
        print("cropping")
        for frame in batch:
            small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
            person_locations = face_recognition.face_locations(small_frame)
            print("Person detect: {}".format(len(person_locations)))
            if(len(person_locations)):
                people, areas = Utils.setDictionary(person_locations)
                # encuentra la cara mas grande
                indexMax = areas.index(max(areas))
                person_location = person_locations[indexMax]
                person_location = list(np.array(person_location)*4)
                top = 

                crop_img = frame[: person_location[0]+person_location[2]-person_location[0] +
                                 0, person_location[3]-0:person_location[3]+person_location[1]-person_location[3]+0]
                crop_img = cv2.cvtColor(crop_img, cv2.COLOR_BGR2RGB)
                faces.append(crop_img)

        # fig = plt.figure(figsize=(8, 8))
        # columns = 2
        # rows = 5
        # for i in range(0, columns*rows):
        #     print("plot {}".format(i))
        #     img = faces[i]
        #     fig.add_subplot(rows, columns, i+1)
        #     plt.imshow(img)
        # plt.show()
        return faces


class Utils:

    def sortDictionary(self, val):
        return val['faceRectangle']['width']

    def setDictionary(self, locations):
        people = list()
        areas = list()
        for face_location in locations:
            width = face_location[1]-face_location[3]
            height = face_location[2]-face_location[0]
            dictionary_of_features = {'faceId': None, 'faceRectangle': {'width': int(width), 'top': int(
                face_location[0]), 'height': int(height), 'left': int(face_location[3])}, 'faceAttributes': None}
            people.append(dictionary_of_features)
            areas.append(width*height)
        return people, areas


# while 1:
#     cv2.imshow('frame', frame)
#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         break
# print(neural.genderPredictor(frame))
# print(neural.agePredictor(frame))
if __name__ == "__main__":

    batch = list()
    Utils = Utils()
    print("cropping")
    cap = cv2.VideoCapture(0)
    # frame = cv2.imread('../Resources/vieja2.jpg')
    for i in range(10):
        print "capture image {}".format(str(i))
        error, frame = cap.read()
        frame = cv2.GaussianBlur(frame, (5, 5), 0)
        batch.append(frame)

    print(batch[0])
    print(batch[0].shape)
    cap.release()

    neural = NeuralClass(batch)

    gender = neural.genderPredictor()
    age = neural.agePredictor()
    print(age)
    gender = neural.genderPredictor()
    print(gender)

    # gender_counter = Counter(gender)
    # print("predict: Gender {} {}%".format(predict["age"]["range"],predict["age"]["percent"]))
    # print("predict: Age {} {}%".format(predict["gender"]["range"],predict["gender"]["percent"]))
    # print(gender_counter.most_common()[0])
    # print(age_counter)
    # print(age)
    # print(gender)
