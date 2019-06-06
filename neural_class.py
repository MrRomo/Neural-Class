
import face_recognition
import cv2
import numpy as np
from collections import Counter


class NeuralClass:

    def __init__(self, frame, percent=1, tolerance=0.6):
        self.frame = frame
        self.percent = percent
        self.tolerance = tolerance
        self.coord = list()
        self.percents = list()
        # inicializa la clase recortando, guardando las caras y descartando los frames malos
        self.faces = self.cropper()

    def cropper(self):
        faces = list()
        frames = list()
        utils = Utils()
        print("cropping")
        for frame in self.frame:
            frame_area = frame.shape[0]*frame.shape[1]
            small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
            person_loc = face_recognition.face_locations(small_frame)
            print("Person detect: {}".format(len(person_loc)))
            if(len(person_loc)):  # detecta si hay personas
                people, areas = utils.setDictionary(person_loc)
                # ordena las caras de mayor a menor detectadas en el frame
                people.sort(key=utils.sortDictionary, reverse=True)
                people = people[0]
                # encuentra la cara mas grande
                face_area = max(areas)
                indexMax = areas.index(face_area)
                person_location = person_loc[indexMax]
                # top, rigth, bottom, left (t,r,b,l)
                t, r, b, l = utils.increase(list(np.array(person_location)*4))
                percent = face_area*100/float(frame_area)
                if(percent >= self.percent):
                    # recortar imagenes image[y:y+h, x:x+w]
                    faces.append(frame[t:b, l:r])
                    frames.append(frame)
                    self.coord.append((t, r, b, l))
                    self.percents.append(round(percent, 2))
        # guarda unicamente los frames donde hay caras
        self.frame = frames
        return faces

    def detect(self):
        pass

    def encode(self):
        person_encoding = face_recognition.face_encodings(self.faces[0])
        return person_encoding

    def compare(self, known_faces, personGroup):
        person_encoding = self.encode()
        matches = face_recognition.compare_faces(
            known_faces, person_encoding, tolerance=self.tolerance)

        print("matches", matches)
        if True in matches:
            first_match_index = matches.index(True)
            people = personGroup[first_match_index]
            distance = face_recognition.face_distance(
                [known_faces[first_match_index]], person_encoding)
            people['accuracy'] = 1-distance[0]*self.tolerance
            return people
        else:
            return []

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

    def increase(self, dimentions):
        prop = 0.9
        prop2 = 1.07
        print(dimentions)
        dimentions[0] = int(dimentions[0]*0.55)
        dimentions[1] = int(dimentions[1]*prop2)
        dimentions[2] = int(dimentions[2]*prop2)
        dimentions[3] = int(dimentions[3]*0.92)
        print(dimentions)
        return dimentions
