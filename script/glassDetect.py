from PIL import Image, ImageDraw
import face_recognition
import cv2
import pickle
import numpy as np
import pandas as pd

cap = cv2.VideoCapture(0)

def getModel(path):

    with open(path) as f:
        clf, labels = pickle.load(f)
        return clf, labels

clf, labels = getModel('../Models/race_and_gender_model.pkl')
COLS = ['No eyewear', 'Eyeglasses', 'Sunglasses']


while 1:
    error, image = cap.read()
    # image = cv2.imread('../Resources/Glass/lentes_0.jpg')
    small_frame = cv2.resize(image, (0, 0), fx=0.25, fy=0.25)

    # Find all facial features in all the faces in the image
    face_landmarks_list = face_recognition.face_landmarks(small_frame)
    face_locations = face_recognition.face_locations(small_frame)

    print("I found {} face(s) in this photograph.".format(len(face_landmarks_list)))

    # Create a PIL imagedraw object so we can draw on the picture
    pil_image = Image.fromarray(image)
    d = ImageDraw.Draw(pil_image)

    for face_landmarks in face_landmarks_list:

        # Print the location of each facial feature in this image
        # print face_landmarks["left_eye"]
        # print face_landmarks["right_eye"]
        l = face_landmarks["left_eye"][0]
        r = face_landmarks["right_eye"][3]

        print (l, r)
        ymax = max((l[1], r[1]))
        ymin = min((l[1], r[1]))
        xmax = r[0]
        xmin = l[1]
        # Let's trace out each facial feature in the image with a line!
        eyes = [(ymax, xmax, ymin, xmin)]

        for (top, right, bottom, left)in (face_locations):
            # Scale back up face locations since the frame we detected in was scaled to 1/4 size
            top *= 4
            right *= 4
            bottom *= 4
            left *= 4
        for (etop, eright, ebottom, eleft) in eyes:
            p = int((right-left))
            eleft = int(left*1)
            eright = int(right*1)
            etop = (etop*4)-int(p*0.1)
            ebottom = (ebottom*4)+int(p*0.16)

        face_encodings = face_recognition.face_encodings(image[top:bottom,left:right])
        if len(face_encodings):
            prediction = pd.DataFrame(clf.predict_proba(face_encodings), columns=labels)
            prediction = prediction.loc[:, COLS]
            print str(prediction)

        
        # Draw a box around the face
        witdh = right-left
        height = ebottom-etop
        ojos = image[etop:ebottom, eleft:eright]
        result = Image.new("RGB", (image.shape[1], image.shape[0]))
        ojos = Image.fromarray(ojos)
        image = Image.fromarray(image)
        w, h = image.size
        coord = (0, 0, w, h)
        result.paste(image, coord)
        w, h = ojos.size
        coord = (0, 0, w, h)
        result.paste(ojos, coord)
        image = np.asarray(result)
        cv2.rectangle(image, (left, top), (right, bottom), (0, 255, 255), 2)
        font = cv2.FONT_HERSHEY_DUPLEX

        cv2.putText(image, str(prediction["Eyeglasses"])+"\n", (left + 6, bottom - 6),font, 0.6, (255, 255, 255), 1)
        cv2.rectangle(image, (eleft, etop), (eright, ebottom), (0, 0, 255), 1)
        cv2.line(image, (l[0]*4,l[1]*4),(r[0]*4,r[1]*4),  (0, 255, 255), 1, 8)


        print result
        print result.size
        print type(result)
    cv2.imshow('Video', image)

    # Hit 'q' on the keyboard to quit!
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release handle to the webcam
cap.release()
cv2.destroyAllWindows()
# Show the picture
