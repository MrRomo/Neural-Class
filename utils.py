# //======================================================================//
# //  This software is free: you can redistribute it and/or modify        //
# //  it under the terms of the GNU General Public License Version 3,     //
# //  as published by the Free Software Foundation.                       //
# //  This software is distributed in the hope that it will be useful,    //
# //  but WITHOUT ANY WARRANTY; without even the implied warranty of      //
# //  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE..  See the      //
# //  GNU General Public License for more details.                        //
# //  You should have received a copy of the GNU General Public License   //
# //  Version 3 in the file COPYING that came with this distribution.     //
# //  If not, see <http://www.gnu.org/licenses/>                          //
# //======================================================================//
# //                                                                      //
# //      Copyright (c) 2019 SinfonIA Pepper RoboCup Team                 //
# //      Sinfonia - Colombia                                             //
# //      https://sinfoniateam.github.io/sinfonia/index.html              //
# //                                                                      //
# //======================================================================//

import cv2
from cv_bridge import CvBridge
import os
import sys


class Utils:

    def __init__(self, source, percent_of_face):
        self.source = source
        self.percent_of_face = percent_of_face
        self.bridge = CvBridge()

    def setProps(self, people, frame_size):
        props = []
        prop = 0
        for face_detected in people:
            pi = (face_detected['faceRectangle']['left'],
                  face_detected['faceRectangle']['top'])
            pf = (face_detected['faceRectangle']['left']+face_detected['faceRectangle']['width'],
                  face_detected['faceRectangle']['top']+face_detected['faceRectangle']['height'])
            prop = (face_detected['faceRectangle']['width']
                    * face_detected['faceRectangle']['height'])
            # guarda el calculo de las proporciones en cada ciclo
            props.append({"pi": pi, "pf": pf, "prop": round(
                prop*100/float(frame_size), 4)})
        return props

    def take_picture_source(self):
        source = self.source
        print("take picture from source: *{}*".format(source))
        if source == 'webcam':
            cap = cv2.VideoCapture(0)
            ret, frame = cap.read()
            frame = cv2.GaussianBlur(frame, (5, 5), 0)
            cap.release()
        elif source == 'file':
            ROOT_PATH = os.path.dirname(sys.modules['__main__'].__file__)
            frame = cv2.imread(ROOT_PATH+"/Resources/gente1.jpg")
        else:
            rospy.wait_for_service("sIA_take_picture")
            takePicture = rospy.ServiceProxy("sIA_take_picture", TakePicture)
            imageRos = takePicture("Take Picture", [0, 2, 11, 30]).response
            frame = self.bridge.imgmsg_to_cv2(imageRos, "bgr8")
        return frame

    def add_features_to_image(self, frame, people):
        frame_size = frame.shape[0]*frame.shape[1]
        percent = []
        isInFront = False
        if people:
            font = cv2.FONT_HERSHEY_SIMPLEX
            props = self.setProps(people, frame_size)
            for prop in props:
                cv2.rectangle(frame, prop['pi'], prop['pf'], (0, 255, 0), 3)
                cv2.putText(frame, str(
                    prop['prop']), prop['pi'], font, 1, (255, 150, 0), 2, cv2.LINE_AA)

            if 'name' in people[0]:
                print(props)
                cv2.rectangle(frame, (props[0]['pi'][0], props[0]['pf']
                                      [1]+50), (props[0]['pf']), (0, 0, 255), cv2.FILLED)
                cv2.putText(frame, str(people[0]['name']), (props[0]['pi'][0]+10,
                                                            props[0]['pf'][1]+30), font, 1, (255, 150, 0), 2, cv2.LINE_AA)
            else:
                cv2.putText(frame, str(
                    props[0]["prop"])+'%', props[0]['pi'], font, 1, (255, 150, 0), 2, cv2.LINE_AA)

            if props[0]['prop'] > self.percent_of_face:
                isInFront = True
                # Remarca la cara mayor
                cv2.rectangle(frame, props[0]['pi'],
                              props[0]['pf'], (0, 0, 255), 5)
        response = {"frame": self.bridge.cv2_to_imgmsg(
            frame, "bgr8"), "isInFront": isInFront}
        return response
    
    def sortDictionary(val):
        return val['faceRectangle']['width']


    def setDictionary(locations):
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

