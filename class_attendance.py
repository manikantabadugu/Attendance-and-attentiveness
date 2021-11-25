import face_recognition
import cv2
from datetime import datetime
import numpy as np
import os
import functions as m

path = 'images of students'
student_names = []
images = []

names_list = os.listdir(path)
for name in names_list:
    current_pic = cv2.imread(f'{path}/{name}')
    images.append(current_pic)
    student_names.append(name.split('.')[0])
print(student_names)



encode_list = m.encodings(images)
print(len(encode_list))




cap = cv2.VideoCapture(0)
while True:
    cam, img= cap.read()
    img_reduced= cv2.resize(img, (0,0), None,0.25,0.25)
    img_reduced = cv2.cvtColor(img_reduced, cv2.COLOR_BGR2RGB)

    faces_loc = face_recognition.face_locations(img_reduced)

    current_encode = face_recognition.face_encodings(img_reduced, faces_loc)

    for enc,faces in zip(current_encode,faces_loc):
        results = face_recognition.compare_faces(encode_list, enc)
        distance = face_recognition.face_distance(encode_list, enc)
        match = np.argmin(distance)

        if results[match]:
            name = student_names[match].upper()
            y1,x2,y2,x1 = faces
            y1, x2, y2, x1= y1*4,x2*4,y2*4,x1*4
            cv2.rectangle(img,(x1,y1),(x2,y2),(255,0,0),2)
            cv2.rectangle(img,(x1,y2-30),(x2,y2),(255,0,0),cv2.FILLED)
            cv2.putText(img, name,(x1+6,y2),cv2.FONT_HERSHEY_COMPLEX,1,(255,255,255),1)
            m.class_attendance(name)
        else:
            cv2.rectangle(img,(x1,y1),(x2,y2),(255,0,0),2)
            cv2.rectangle(img, (x1, y2 - 30), (x2, y2), (255, 0, 0), cv2.FILLED)
            cv2.putText(img,'unKNOWN', (x1 + 6, y2), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 255), 1)

    cv2.imshow('cam',img)
    cv2.waitKey(1)








