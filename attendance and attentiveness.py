import face_recognition
import cv2 as cv
from datetime import datetime
import numpy as np
import os
import functions as m
import time
import winsound

path = 'images of students'
student_names = []
images = []
TOTAL_BLINKS = 0
COUNTER = 0
CLOSED_EYES_FRAME=0
optimal_blinks= 25
t_end = time.time() + 60
print(time.time())
print(t_end)

names_list = os.listdir(path)

#printing student names in class
for name in names_list:
    current_pic = cv.imread(f'{path}/{name}')
    images.append(current_pic)
    student_names.append(name.split('.')[0])
print(student_names)

#printing total number of faces recognized from the given inputs
encode_list = m.encodings(images)
print(len(encode_list))

cap = cv.VideoCapture(0)
while True:
    cam, img= cap.read()
    img_reduced= cv.resize(img, (0,0), None,0.25,0.25)
    img_reduced = cv.cvtColor(img_reduced, cv.COLOR_BGR2RGB)

    faces_loc = face_recognition.face_locations(img_reduced)

    current_encode = face_recognition.face_encodings(img_reduced, faces_loc)

    for enc,faces in zip(current_encode,faces_loc):
        results = face_recognition.compare_faces(encode_list, enc)
        distance = face_recognition.face_distance(encode_list, enc)
        match = np.argmin(distance)
        y1, x2, y2, x1 = faces
        y1, x2, y2, x1 = y1*4, x2*4, y2*4, x1*4

        if results[match]:
            name = student_names[match].upper()
            y1,x2,y2,x1 = faces
            y1, x2, y2, x1= y1*4,x2*4,y2*4,x1*4
            #cv.rectangle(img,(x1,y1),(x2,y2),(255,0,0),2)
            cv.rectangle(img,(x1,y2-30),(x2,y2),(255,0,0),cv.FILLED)
            cv.putText(img, name,(x1+6,y2),cv.FONT_HERSHEY_COMPLEX,1,(255,255,255),1)
            m.class_attendance(name)

            grayFrame = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
            height, width = grayFrame.shape
            circleCenter = (int(width / 2), 50)
            # calling the face detector funciton
            image, face = m.faceDetector(img, grayFrame)

            if face is not None:

                # calling landmarks detector function.
                image, PointList = m.faceLandmakDetector(img, grayFrame, face, False)
                # print(PointList)

                # cv.putText(frame, f'FPS: {round(FPS,1)}',
                # (460, 20), m.fonts, 0.7, m.YELLOW, 2)
                RightEyePoint = PointList[36:42]
                LeftEyePoint = PointList[42:48]
                leftRatio, topMid, bottomMid = m.blinkDetector(LeftEyePoint)
                rightRatio, rTop, rBottom = m.blinkDetector(RightEyePoint)

                blinkRatio = (leftRatio + rightRatio) / 2
                # cv.circle(image, circleCenter, (int(blinkRatio * 4.3)), m.CHOCOLATE, -1)
                # cv.circle(image, circleCenter, (int(blinkRatio * 3.2)), m.CYAN, 2)
                # cv.circle(image, circleCenter, (int(blinkRatio * 2)), m.GREEN, 3)

                if blinkRatio > 4:
                    COUNTER += 1
                    cv.putText(image, f'Blink', (70, 50),
                               m.fonts, 0.8, m.LIGHT_BLUE, 2)

                    # print("blink")
                else:
                    if COUNTER > CLOSED_EYES_FRAME:
                        TOTAL_BLINKS += 1
                        COUNTER = 0
                cv.putText(image, f'Total Blinks: {TOTAL_BLINKS}', (230, 17),
                           m.fonts, 0.5, m.ORANGE, 2)
                cv.putText(image, f'optimal blinks:{optimal_blinks}', (430, 20), m.fonts, 0.5, m.RED, 1)
                cv.putText(image, f'time:{t_end}', (430, 50), m.fonts, 0.5, m.RED, 1)
                if time.time() > t_end:
                    if TOTAL_BLINKS < optimal_blinks:

                        cv.putText(image, f'some thing wrong', (30, 20),
                                   m.fonts, 0.5, m.RED, 1)
                        frequency = 2500  # Set Frequency To 2500 Hertz
                        duration = 1500  # Set Duration To 1000 ms == 1 second
                        winsound.Beep(frequency, duration)
                        # time.sleep(2)
                        # optimal_blinks += 5
                        t_end = time.time() + 60
                        TOTAL_BLINKS = 0
                    else:
                        # optimal_blinks += 5
                        t_end = time.time() + 60
                        TOTAL_BLINKS = 0


        else:
            #cv.rectangle(img,(x1,y1),(x2,y2),(255,0,0),2)
            cv.rectangle(img, (x1, y2 - 30), (x2, y2), (255, 0, 0), cv.FILLED)
            cv.putText(img,'unKNOWN', (x1 + 6, y2), cv.FONT_HERSHEY_COMPLEX, 1, (255, 255, 255), 1)







    cv.imshow('cam',img)
    #cv.imshow('gray',grayFrame)
    cv.waitKey(1)



