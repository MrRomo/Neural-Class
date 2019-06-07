
import sys
import os
import face_recognition
import cv2
import numpy as np
from collections import Counter


class NeuralClass:

    def __init__(self, frame, percent=0.1, tolerance=0.4):
        self.ROOT_PATH = os.path.dirname(sys.modules['__main__'].__file__)
        self.frame = frame
        self.percent = percent
        self.tolerance = tolerance
        self.MODEL_MEAN_VALUES = (78.4263377603, 87.7689143744, 114.895847746)
        self.age_list = ['(0, 2)', '(4, 6)', '(8, 12)', '(15, 20)','(25, 32)', '(38, 43)', '(48, 53)', '(60, 100)']
        self.coord = []
        self.percents = []
        self.people = []
        self.utils = Utils()
        self.age_net = self.load_caffe_models()
        self.faces = self.cropper()# inicializa la clase recortando, guardando las caras y descartando los frames malos

    def load_caffe_models(self):
        age_net = cv2.dnn.readNetFromCaffe(
            self.ROOT_PATH + '/Models/deploy_age.prototxt', self.ROOT_PATH+'/Models/age_net.caffemodel')
        return age_net

    def cropper(self):
        faces = list()
        frames = list()
        print("cropping")
        for frame in self.frame:
            frame_area = frame.shape[0]*frame.shape[1]
            small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
            person_loc = face_recognition.face_locations(small_frame)
            print("Person detect: {} {} {} ".format(len(person_loc),person_loc[0],type(person_loc[0])))
            if(len(person_loc)):  # detecta si hay personas
                people, areas = self.utils.setDictionary(person_loc)
                # ordena las caras de mayor a menor detectadas en el frame
                people.sort(key=self.utils.sortDictionary, reverse=True)
                print("people raw", people)
                people = people[0]
                # encuentra la cara mas grande
                face_area = max(areas)
                indexMax = areas.index(face_area)
                person_location = person_loc[indexMax]
                print("dim",person_location)
                # top, rigth, bottom, left (t,r,b,l)
                t, r, b, l = self.utils.increase(person_location)
                percent = face_area*100/float(frame_area)
                print("percent {}%".format(percent))
                print("dim", t,r,b,l)
                if(percent >= self.percent):
                    # recortar imagenes image[y:y+h, x:x+w]
                    print("people cropper", people)
                    faces.append(frame[t:b, l:r])
                    frames.append(frame)
                    self.coord.append((t, r, b, l))
                    self.percents.append(round(percent, 2))
                    self.people.append(people)
        # guarda unicamente los frames donde hay caras
        self.frame = frames
        return faces

    def detect(self):
        if len(self.people):
            return [self.people[0]]
        return []

    def encode(self):
        if len(self.detect()):
            print self.coord[0], type(self.coord[0])
            person_encoding = face_recognition.face_encodings(self.frame[0],[self.coord[0]])[0]
            print "person emcoding",person_encoding
            return person_encoding
        else:
            []

    def compare(self, known_faces, personGroup):
        if len(self.detect()):
            person_encoding = self.encode()
            print(known_faces)
            print type(known_faces)
            print(person_encoding)
            print type(person_encoding)
            matches = face_recognition.compare_faces(known_faces, person_encoding, tolerance=self.tolerance)

            print("matches", matches)
            print("distance : ", face_recognition.face_distance(known_faces,person_encoding))
            if True in matches:
                first_match_index = matches.index(True)
                people = personGroup[first_match_index]
                distance = face_recognition.face_distance(
                    [known_faces[first_match_index]], person_encoding)
                people['accuracy'] = 1-distance[0]*self.tolerance
                people['faceRectangle'] = self.people[0]['faceRectangle']
                return people
            else:
                return []
        else:
            return []

    def race(self):
        pass

    def glass(self):
        pass

    def age(self):
        if not(len(self.detect())):
            return {"age": None, "percent": None}
        predict = list()
        for frame in self.faces:
            blob = cv2.dnn.blobFromImage(
                frame, 1, (227, 227), self.MODEL_MEAN_VALUES, swapRB=False)
            # Predict Age
            self.age_net.setInput(blob)
            age_preds = self.age_net.forward()
            age = self.age_list[age_preds[0].argmax()]
            predict.append(age)  # predict contiene todas las prediciones
        
        age_counter = Counter(predict)# counter permite conocer el promedio y la moda de las precciones
        result = dict()
        print()
        if(len(predict)):
            result = {
                "age": age_counter.most_common()[0][0],
                "percent": age_counter.most_common()[0][1]/float(len(predict))
            }
        return result

    def hair(self):
        pass

    def gender(self):
        pass


class Utils:

    def sortDictionary(self, val):
        return val['faceRectangle']['width']*val['faceRectangle']['left']

    def setDictionary(self, locations):
        people = list()
        areas = list()
        for face_location in locations:
            print type(face_location)
            face_location =  list(np.array(face_location)*4)
            width = face_location[1]-face_location[3]
            height = face_location[2]-face_location[0]
            dictionary_of_features = {'faceId': None, 'faceRectangle': {'width': int(width), 'top': int(
                face_location[0]), 'height': int(height), 'left': int(face_location[3])}, 'faceAttributes': None}
            people.append(dictionary_of_features)
            areas.append(width*height)
        return people, areas

    def increase(self, dimentions):
        dim = list()
        prop = 1.07
        dim.append(int(dimentions[0]*0.3))
        dim.append(int(dimentions[1]*prop))
        dim.append(int(dimentions[2]*prop))
        dim.append(int(dimentions[3]*0.92))
        print(dim)
        return dim
