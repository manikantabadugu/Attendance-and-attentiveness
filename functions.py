import cv2 as cv
import numpy as np
import dlib
import math
import face_recognition
from datetime import datetime

# variables
fonts = cv.FONT_HERSHEY_COMPLEX

# colors
YELLOW = (0, 247, 255)
CYAN = (255, 255, 0)
MAGENTA = (255, 0, 242)
GOLDEN = (32, 218, 165)
LIGHT_BLUE = (255, 9, 2)
PURPLE = (128, 0, 128)
CHOCOLATE = (30, 105, 210)
PINK = (147, 20, 255)
ORANGE = (0, 69, 255)
GREEN = (0, 255, 0)
LIGHT_GREEN = (0, 255, 13)
LIGHT_CYAN = (255, 204, 0)
BLUE = (255, 0, 0)
RED = (0, 0, 255)
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
LIGHT_RED = (2, 53, 255)


detectFace = dlib.get_frontal_face_detector()
# landmarks detector
predictor = dlib.shape_predictor(
    r"C:\Users\User\PycharmProjects\attendance\venv\facerecognition\Lib\site-packages\face_recognition_models\models\shape_predictor_68_face_landmarks.dat")

def encodings(photo):
    encodings_list = []
    for img in photo:
        img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
        encodings_list.append(face_recognition.face_encodings(img)[0])
    return encodings_list

def class_attendance(name):
    with open('attendance.csv','r+') as f:
        data_in_csv = f.readlines()
        new_name_list = []
        for data in data_in_csv:
            new_name_list.append(data.split(',')[0])
        if name not in new_name_list:
            now = datetime.now()
            timing = now.strftime('%H:%M:%S %D')
            f.writelines(f'\n{name},{timing}')


def midpoint(pts1, pts2):
    x, y = pts1
    x1, y1 = pts2
    xOut = int((x + x1)/2)
    yOut = int((y1 + y)/2)
    # print(xOut, x, x1)
    return (xOut, yOut)

def eucaldainDistance(pts1, pts2):
    x, y = pts1
    x1, y1 = pts2
    eucaldainDist = math.sqrt((x1 - x) ** 2 + (y1 - y) ** 2)

    return eucaldainDist

def faceDetector(image, gray, Draw=True):
    cordFace1 = (0, 0)
    cordFace2 = (0, 0)
    # getting faces from face detector
    faces = detectFace(gray)

    face = None
    # looping through All the face detected.
    for face in faces:
        # getting coordinates of face.
        cordFace1 = (face.left(), face.top())
        cordFace2 = (face.right(), face.bottom())

        # draw rectangle if draw is True.
        if Draw == True:
            cv.rectangle(image, cordFace1, cordFace2, GREEN, 2)
    return image, face


def blinkDetector(eyePoints):
    top = eyePoints[1:3]
    bottom = eyePoints[4:6]
    # finding the mid point of above points
    topMid = midpoint(top[0], top[1])
    bottomMid = midpoint(bottom[0], bottom[1])
    # getting the actual width and height eyes using eucaldainDistance function
    VerticalDistance = eucaldainDistance(topMid, bottomMid)
    HorizontalDistance = eucaldainDistance(eyePoints[0], eyePoints[3])
    # print()

    blinkRatio = (HorizontalDistance/VerticalDistance)
    return blinkRatio, topMid, bottomMid


def faceLandmakDetector(image, gray, face, Draw=True):
    # calling the landmarks predictor
    landmarks = predictor(gray, face)
    pointList = []
    # looping through each landmark
    for n in range(0, 68):
        point = (landmarks.part(n).x, landmarks.part(n).y)
        # getting x and y coordinates of each mark and adding into list.
        pointList.append(point)
        # draw if draw is True.
        if Draw == True:
            # draw circle on each landmark
            cv.circle(image, point, 3, ORANGE, 1)
    return image, pointList

def EyeTracking(image, gray, eyePoints):
    # getting dimensions of image
    dim = gray.shape
    # creating mask .
    mask = np.zeros(dim, dtype=np.uint8)


    # converting eyePoints into Numpy arrays.
    PollyPoints = np.array(eyePoints, dtype=np.int32)
    # Filling the Eyes portion with WHITE color.
    cv.fillPoly(mask, [PollyPoints], 255)
    eyeImage = cv.bitwise_and(gray, gray, mask=mask)

    maxX = (max(eyePoints, key=lambda item: item[0]))[0]
    minX = (min(eyePoints, key=lambda item: item[0]))[0]
    maxY = (max(eyePoints, key=lambda item: item[1]))[1]
    minY = (min(eyePoints, key=lambda item: item[1]))[1]

    eyeImage[mask == 0] = 255

    cropedEye = eyeImage[minY:maxY, minX:maxX]

    height, width = cropedEye.shape

    divPart = int(width / 3)

    ret, thresholdEye = cv.threshold(cropedEye, 100, 255, cv.THRESH_BINARY)

    # dividing the eye into Three parts .
    rightPart = thresholdEye[0:height, 0:divPart]
    centerPart = thresholdEye[0:height, divPart:divPart + divPart]
    leftPart = thresholdEye[0:height, divPart + divPart:width]

    # counting Black pixel in each part using numpy.
    rightBlackPx = np.sum(rightPart == 0)
    centerBlackPx = np.sum(centerPart == 0)
    leftBlackPx = np.sum(leftPart == 0)
    pos, color = Position([rightBlackPx, centerBlackPx, leftBlackPx])
    # print(pos)

    return mask, pos, color

def Position(ValuesList):

    maxIndex = ValuesList.index(max(ValuesList))
    posEye = ''
    color = [WHITE, BLACK]
    if maxIndex == 0:
        posEye = "Right"
        color = [YELLOW, BLACK]
    elif maxIndex == 1:
        posEye = "Center"
        color = [BLACK, MAGENTA]
    elif maxIndex == 2:
        posEye = "Left"
        color = [LIGHT_CYAN, BLACK]
    else:
        posEye = "Eye Closed"
        color = [BLACK, WHITE]
    return posEye, color

