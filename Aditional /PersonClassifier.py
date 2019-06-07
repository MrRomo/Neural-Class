
# coding: utf-8

# In[26]:


import argparse
import os
import face_recognition
import numpy as np
import sklearn
import pickle
from face_recognition import face_locations
from PIL import Image, ImageDraw, ImageFont
from tqdm import tqdm
import cv2
import pandas as pd
import math

# import unicodedata
#cap = cv2.VideoCapture(0)
#leido, frame = cap.read()

frame = cv2.imread('Hombre.jpg')
frame = cv2.imread('Anteojos.png')
frame = cv2.imread('Asiatico.jpg')
frame = cv2.imread('Niño.jpeg')

class PersonClassifier:

    def __init__(self):
        self.frame = frame
        self.race = None   
        self.gender = None
        self.age = None    
        self.ColorHair= None
        self.Glasses= None
        self.COLS = ['Male', 'Asian', 'White', 'Black',  'Baby', 'Child', 'Youth', 'Middle Aged', 'Senior', 'Black Hair', 'Blond Hair', 'Brown Hair', 'Bald', 'No eyewear', 'Eyeglasses', 'Sunglasses', 'Mustache', 'Smiling','Curly Hair', 'Wavy Hair', 'Straight Hair']
        self.N_UPSCLAE = 1
        self.model_path =('race_and_gender_model.pkl') # args.model

                #predecir una imagen
    def predict_one_image(self,img_path, clf, labels):
        print("2. predict_one_image")
        print(" ")

        """Predict face attributes for all detected faces in one image"""
        print("Predecir atributos de cara para todas las caras detectadas en una imagen.")
        
        face_encodings, locs = self.extract_features(img_path)
        #print(face_encodings, locs)
        
        
        if not face_encodings:
            return None, None
        pred = pd.DataFrame(clf.predict_proba(face_encodings),
                            columns = labels)
        #print(pred)
        pred = pred.loc[:, self.COLS]
        #print("Fin pred")
        print ("2.pred",pred)
        print ("2.locs",locs)
        print("_2. End predict_one_image ")
        print(" ")

        return pred, locs
        
    def gender_race(self,frame):
        with open(self.model_path) as f:

            clf, labels = pickle.load(f)
            pred, locs = self.predict_one_image(frame, clf, labels)
            locs =                 pd.DataFrame(locs, columns = ['top', 'right', 'bottom', 'left'])
            df = pd.concat([pred, locs], axis=1)
            #print(df)
            img,gender_race = self.draw_attributes(frame, df)
        return gender_race
        print("0. gender_race")   
        #diccionario= {'Genero':None,'raza':None}
        
    #extraer_características
    def extract_features(self,img_path):
        print("3. extraer_características")
        print(" ")
        """Exctract 128 dimensional features"""
        print("Extraer 128 características dimensionales.")
        
        locs = face_locations(img_path, number_of_times_to_upsample = self.N_UPSCLAE)
        if len(locs) == 0:
            return None, None
        face_encodings = face_recognition.face_encodings(frame, known_face_locations=locs)
        #print ("face_encodings",face_encodings)
        #print ("locs",locs)
        print("_3. End extract_features")

        return face_encodings, locs
 
    #dibujar atributos
    def draw_attributes(self,img_path, df):
        print(" ")
        print("4. dibujar atributos")
        print(" ")

        """Write bounding boxes and predicted face attributes on the image"""
        print("Escribe cuadros delimitadores y atributos de cara predichos en la imagen.")
        
        img = img_path 

        for row in df.iterrows():
            caracteristicas=len(row[1])
            Car=caracteristicas-4
            print(row[1][Car:].astype(int))
            top, right, bottom, left = row[1][Car:].astype(int)
              
            print("row")
            print(type(row)) 
            print(row)
            print("")
            self.setGender(row[1]['Male'])
            self.setRace(np.argmax(row[1][1:4]))
            self.setAge(np.argmax(row[1][4:9]))
            self.setColorHair(np.argmax(row[1][9:13]))
            print("Asasasas")
            print(type(row[1][13]))
            self.setGlasses(row[1][13],np.argmax(row[1][13:16]))
            text_showed = "{} {}".format(self.race, self.gender)
            #print (text_showed)
            Features = {'Gender':self.gender,'Race':self.race,'Age':self.setAge,'ColorHair':self.ColorHair,'Glasses':self.Glasses}
            cv2.rectangle(img, (left, top), (right, bottom), (0, 0, 255), 2)
            font = cv2.FONT_HERSHEY_DUPLEX
            img_width = img.shape[1]
            cv2.putText(img, text_showed, (left + 6, bottom - 6), font, 0.5, (255, 255, 255), 1)
            #print ("img",img)
            print ("4. gender_race",gender_race)
            print("_4. End draw_attributes ")
            print(" ")

        return img,gender_race
    
    def setGender(self,EstGender):

        if EstGender >= 0.5:
            self.gender = 'Male'
        else:
            self.gender = 'Female'

    def getGender(self):
        return self.gender

    def setRace(self,EstRace):
        self.race = EstRace

    def getRace(self):
        return self.race

    def setAge(self,EstAge):
        self.age = EstAge

    def getAge(self):
        return self.age

    def setColorHair(self,EstColorHair):
        self.ColorHair= EstColorHair

    def getColorHair(self):
        return self.ColorHair   
    
    def setGlasses(self,NoEyewear,EstGlasses):
        if math.isnan(NoEyewear):
            self.EstGlasses= 'No Eyewear'
        else: 
            self.EstGlasses= EstGlasses

    def getGlasses(self):
        return self.EstGlasses   


PersonC = PersonClassifier()    
gender_race = PersonC.gender_race(frame)
print('Gender:', PersonC.getGender() ,'race:',PersonC.getRace())
print('Age:',PersonC.getAge(),'ColorHair:',PersonC.getColorHair())
print('Glasses:',PersonC.getGlasses())
