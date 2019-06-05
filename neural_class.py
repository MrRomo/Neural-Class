
import face_recognition
import cv2
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter


class NeuralClass:

    def __init__(self, frame):
        self.frame = frame
        self.faces = self.cropper()

    def cropper(self):
        faces = list()
        utils = Utils()
        print("cropping")
        for frame in self.frame:
            small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
            person_loc = list(np.array(face_recognition.face_locations(small_frame))*4)
            print("Person detect: {}".format(len(person_loc)))
            if(len(person_loc)):
                people, areas = utils.setDictionary(person_loc)
                # encuentra la cara mas grande
                indexMax = areas.index(max(areas))
                person_location = person_loc[indexMax]
                person_location = list(np.array(person_location)*4)
                faces.append({"frame": frame, "cord": person_location})

        return faces

    def detect(self):
        pass

    def encode(self):
        pass

    def compare(self):
        pass

    def race(self):
        pass

    def glass(self):
        pass

    def age(self):
        pass

    def hair(self):
        pass

    def gender(self):
        pass



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