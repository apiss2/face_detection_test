import os
import sys
import cv2
import numpy
from PIL import Image
from mtcnn.detector import detect_faces

import time

camera_id = 0
cap = cv2.VideoCapture(camera_id)
if not cap.isOpened():
    sys.exit()

while True:
    ret, frame = cap.read()

    image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    pilImg = Image.fromarray(numpy.uint8(image))
    boxes, landmarks = detect_faces(pilImg)

    for box in boxes:
        box = [int(i) for i in box]
        image = cv2.rectangle(image, (box[0], box[1]), (box[2], box[3]), (255, 0, 0), 2)

    for marks in landmarks:
        for i,j in zip(marks[:5],marks[5:]):
            image = cv2.circle(image, (int(i),int(j)), 3, color=(255, 0, ),thickness=2)

    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    cv2.imshow("window", image)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
