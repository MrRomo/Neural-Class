
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

# import unicodedata
cap = cv2.VideoCapture(0)
leido, frame = cap.read()


def gender_race(frame):
    COLS = ['Male', 'Asian', 'White', 'Black']
    N_UPSCLAE = 1
    #diccionario= {'Genero':None,'raza':None}
    def extract_features(img_path):
        """Exctract 128 dimensional features
        """
        locs = face_locations(img_path, number_of_times_to_upsample = N_UPSCLAE)
        if len(locs) == 0:
            return None, None
        face_encodings = face_recognition.face_encodings(frame, known_face_locations=locs)
        return face_encodings, locs


    def predict_one_image(img_path, clf, labels):
        """Predict face attributes for all detected faces in one image
        """
        face_encodings, locs = extract_features(img_path)
        if not face_encodings:
            return None, None
        pred = pd.DataFrame(clf.predict_proba(face_encodings),
                            columns = labels)
        pred = pred.loc[:, COLS]
        return pred, locs

    def draw_attributes(img_path, df):
        """Write bounding boxes and predicted face attributes on the image
        """
        img = img_path #cv2.imread(img_path)
        # img  = cv2.cvtColor(color, cv2.COLOR_BGR2RGB)
        for row in df.iterrows():
            top, right, bottom, left = row[1][4:].astype(int)
            if row[1]['Male'] >= 0.5:
                gender = 'Male'
            else:
                gender = 'Female'

            race = np.argmax(row[1][1:4])
            text_showed = "{} {}".format(race, gender)
            #print (text_showed)
            gender_race= {'Gender':gender,'race':race}
            cv2.rectangle(img, (left, top), (right, bottom), (0, 0, 255), 2)
            font = cv2.FONT_HERSHEY_DUPLEX
            img_width = img.shape[1]
            cv2.putText(img, text_showed, (left + 6, bottom - 6), font, 0.5, (255, 255, 255), 1)

        return img,gender_race


    #def main():
    #   parser = argparse.ArgumentParser()

    model_path =('race_and_gender_model.pkl') # args.model


    with open(model_path) as f:

        clf, labels = pickle.load(f)
        pred, locs = predict_one_image(frame, clf, labels)
        locs = \
            pd.DataFrame(locs, columns = ['top', 'right', 'bottom', 'left'])
        df = pd.concat([pred, locs], axis=1)
        #print(df)
        img,gender_race = draw_attributes(frame, df)
    return gender_race


gender_race=gender_race(frame)

print(gender_race)

   



