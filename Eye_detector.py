import cv2 as cv 
import functions as m
import time
import winsound


TOTAL_BLINKS = 0
COUNTER = 0
CLOSED_EYES_FRAME=0
optimal_blinks= 25
t_end = time.time() + 20
print(time.time())
print(t_end)

camera = cv.VideoCapture(0)
while True:
    #FRAME_COUNTER += 1
    # getting frame from camera
    ret, frame = camera.read()
    if ret == False:
        break

    # converting frame into Gry image.
    grayFrame = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    height, width = grayFrame.shape
    circleCenter = (int(width/2), 50)
    # calling the face detector funciton
    image, face = m.faceDetector(frame, grayFrame)

    if face is not None:

        # calling landmarks detector funciton.
        image, PointList = m.faceLandmakDetector(frame, grayFrame, face, False)
        # print(PointList)

        #cv.putText(frame, f'FPS: {round(FPS,1)}',
                   #(460, 20), m.fonts, 0.7, m.YELLOW, 2)
        RightEyePoint = PointList[36:42]
        LeftEyePoint = PointList[42:48]
        leftRatio, topMid, bottomMid = m.blinkDetector(LeftEyePoint)
        rightRatio, rTop, rBottom = m.blinkDetector(RightEyePoint)

        blinkRatio = (leftRatio + rightRatio) / 2
        #cv.circle(image, circleCenter, (int(blinkRatio * 4.3)), m.CHOCOLATE, -1)
        #cv.circle(image, circleCenter, (int(blinkRatio * 3.2)), m.CYAN, 2)
        #cv.circle(image, circleCenter, (int(blinkRatio * 2)), m.GREEN, 3)

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
        #cv.putText(image, f'time:{t_end.strftime("%H:%M:%S")}', (430, 50), m.fonts, 0.5, m.RED, 1)
        if time.time() > t_end:
            if TOTAL_BLINKS < optimal_blinks:


                cv.putText(image, f'some thing wrong', (30, 20),
                           m.fonts, 0.5, m.RED, 1)
                frequency = 2500  # Set Frequency To 2500 Hertz
                duration = 1500  # Set Duration To 1000 ms == 1 second
                winsound.Beep(frequency, duration)
                #time.sleep(2)
                #optimal_blinks += 5
                t_end = time.time()+60
                TOTAL_BLINKS = 0
            else:
                #optimal_blinks += 5
                t_end = time.time()+60
                TOTAL_BLINKS = 0








    cv.imshow('cam', frame)
    cv.waitKey(1)

