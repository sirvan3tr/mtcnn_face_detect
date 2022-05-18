import numpy as np
import cv2
from mtcnn_cv2 import MTCNN

cap = cv2.VideoCapture('face.mp4')

if cap.isOpened() == False:
    print('Cannot open the file')

detector = MTCNN()

while True:
    ret, frame = cap.read()
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    result = detector.detect_faces(frame)

    if len(result) > 0:
        for face in result:
            keypoints = face['keypoints']
            # Place a rectangle around detected faces
            cv2.rectangle(frame, face['box'], (0, 155, 255), 0)
            # Place dots on face features
            for key, face_feature in face['keypoints'].items():
                cv2.circle(frame, face_feature, 2, (0,155,255), 2)

    if ret == True:
        cv2.imshow('Frame', frame)

        if cv2.waitKey(25) & 0xFF == 27:
            break
    else:
        break

cap.release()
cap.destroyAllWindow()
