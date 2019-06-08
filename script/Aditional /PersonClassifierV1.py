# coding: utf-8


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


class PersonClassifier:

    def __init__(self):
        self.frame = frame
        self.race = None
        self.gender = None
        self.age = None
        self.ColorHair = None
        self.Glasses = None
        self.COLS = ['Male', 'Asian', 'White', 'Black',  'Baby', 'Child', 'Youth', 'Middle Aged', 'Senior', 'Black Hair', 'Blond Hair',
            'Brown Hair', 'Bald', 'No eyewear', 'Eyeglasses', 'Sunglasses', 'Mustache', 'Smiling', 'Curly Hair', 'Wavy Hair', 'Straight Hair']
        self.N_UPSCLAE = 1
        self.model_path = ('race_and_gender_model.pkl')  # args.model

                # predecir una imagen
    def predict_one_image(self, img_path, clf, labels):

        face_encodings, locs = self.extract_features(img_path)
        if not face_encodings:
                return None, None
            pred = pd.DataFrame(clf.predict_proba(face_encodings),columns = labels)
            pred = pred.loc[:, self.COLS]
            return pred, locs
        
    def gender_race(self,frame):
        with open(self.model_path) as f:

            clf, labels = pickle.load(f)
            pred, locs = self.predict_one_image(frame, clf, labels)
            locs =                 pd.DataFrame(locs, columns = ['top', 'right', 'bottom', 'left'])
            df = pd.concat([pred, locs], axis=1)
            img,gender_race = self.draw_attributes(frame, df)
        return gender_race
    
    def extract_features(self,img_path):
       locs = face_locations(img_path, number_of_times_to_upsample = self.N_UPSCLAE)
        if len(locs) == 0:
            return None, None
        face_encodings = face_recognition.face_encodings(self.faces[i])
    
        return face_encodings, locs
 
    # dibujar atributos
    def draw_attributes(self,img_path, df):
       
        img = img_path 
        for row in df.iterrows():
            caracteristicas=len(row[1])
            Car=caracteristicas-4
            top, right, bottom, left = row[1][Car:].astype(int)
            self.setGender(row[1]['Male'])
            self.setRace(np.argmax(row[1][1:4]))
            self.setAge(np.argmax(row[1][4:9]))
            self.setColorHair(np.argmax(row[1][9:13]))
            self.setGlasses(row[1][13],np.argmax(row[1][13:16]))
            text_showed = "{} {}".format(self.race, self.gender)
            Features = {'Gender':self.gender,'Race':self.race,'Age':self.age,'ColorHair':self.ColorHair,'Glasses':self.Glasses}
            cv2.rectangle(img, (left, top), (right, bottom), (0, 0, 255), 2)
            font = cv2.FONT_HERSHEY_DUPLEX
            img_width = img.shape[1]
            cv2.putText(img, text_showed, (left + 6, bottom - 6), font, 0.5, (255, 255, 255), 1)

        return img,Features
    
    def q(self,EstGender):

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
            self.Glasses= 'No Eyewear'
        else: 
            self.Glasses= EstGlasses

    def getGlasses(self):
        return self.Glasses
    
PersonC=PersonClassifier()    
Features = PersonC.gender_race(frame)

def ClasificadorDeFotos(tam):
    intro='p'
    ext='.jpg'
    cont=0
    dfList=list()
    tam=7
    for index in range(tam):
        print(" ")
        print("ENTRENAMIENTO ", index+1)
        print(" ")
        nombre = intro + str(index+1) + ext
        PersonC=PersonClassifier()    
        img=cv2.imread(nombre,3)
        Features = PersonC.gender_race(img)
        Features['Nombre']=nombre
        print("Final Process")
        print(" ")
        df = pd.DataFrame([[Features[key] for key in Features.keys()]], columns= Features.keys())
        dfList.append(df)
        print(" ")
    df = pd.concat(dfList)
    return df,Features.keys()


df,keys=ClasificadorDeFotos(7)


df

keys

[Features[key] for key in Features.keys()]


def EscogerMejor(df,keys):
    Dic=dict()
    for key in keys:
        if key !='Nombre':
            Col = [key,'Nombre']
            print(Col)
            param = df[Col].groupby([key]).count()
            param = param.sort_values(by='Nombre', ascending=False)
            param = param.index
            Dic[key]= param[0]
    return Dic

df.groupby(['Age']).count()
Age=df[['Age','Nombre']].groupby(['Age']).count()
Age=Age.sort_values(by='Nombre', ascending=False)
Age=Age.index
Age[0]
df[['Age','Nombre']].groupby(['Age']).max()
df[['Age']].count()

