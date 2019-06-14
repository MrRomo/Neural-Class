from PIL import Image, ImageDraw
import face_recognition
import cv2
import numpy as np
import keras
import tensorflow as tf
from io import BytesIO
from keras.applications.inception_v3 import InceptionV3, decode_predictions
from keras import backend as K
from keras.preprocessing import image

# Load the jpg file into a numpy array

iv3 = InceptionV3()
cap = cv2.VideoCapture(0)


def getGlass(face):
    sunglass = 'No'
    print (face)
    print type(face)
    face = cv2.resize(face, dsize=(299, 299), interpolation=cv2.INTER_CUBIC)
    face = face.astype('float64')
    face /= 255  # rango entre 0 y 1
    face -= 0.5  # Le restamos 0.5 para que los valores bajos pasen a ser negativos, para posteriormente mulitplicarlos
    face *= 2  # Lo que fina
    print(face.shape)
    print("""Alto : {}, Ancho : {}, y profundidad de los pifaceeles al ser RGB son : {}""".format(
        face.shape[0], face.shape[1], face.shape[2]))
    face = face.reshape([1, face.shape[0], face.shape[1], face.shape[2]])
    face.shape
    pred = iv3.predict(face)
    glass = {'sunglass': pred[0][836], 'sunglasses': pred[0]
             [837], 'binoculars': pred[0][447]}
    print("Glass pred {}".format(glass))
    if (max(glass.values()) > 0.2):
        sunglass = "Yes"
    print(sunglass)
    return sunglass


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

        print face_locations
        print type(face_locations)
        print eyes
        print type(eyes)
        # Draw a box around the face
        witdh = right-left
        height = ebottom-etop
        ojos = image[etop:ebottom, eleft:eright]
        result = Image.new("RGB", (image.shape[1], image.shape[0]))
        pred = getGlass(ojos)
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
        cv2.putText(image, pred+" tiene gafas", (left + 6, bottom - 6),font, 0.6, (255, 255, 255), 1)
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
