
import face_recognition
import cv2
import numpy as np
from collections import Counter


class NeuralClass:

    def __init__(self, frame):
        self.frame = frame
        self.coord = list()
        self.faces = self.cropper()


    def cropper(self):
        faces = list()
        utils = Utils()
        print("cropping")
        for frame in self.frame:
            small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
            person_loc = face_recognition.face_locations(small_frame)
            print("Person detect: {}".format(len(person_loc)))
            if(len(person_loc)):
                people, areas = utils.setDictionary(person_loc)
                people.sort(key=utils.sortDictionary, reverse=True)
                people = people[0]
                # encuentra la cara mas grande
                indexMax = areas.index(max(areas))
                person_location = person_loc[indexMax]
                t, r, b, l = list(np.array(person_location)*4)
                crop_img = frame[t:t+(r-l), l:l+(b-t)]
                crop_img = cv2.cvtColor(crop_img, cv2.COLOR_BGR2RGB)
                faces.append(crop_img)
                self.coord.append((t, r, b, l))
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
