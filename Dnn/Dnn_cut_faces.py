














from imutils.video import VideoStream
import numpy as np
import argparse
import imutils
import time
import cv2

proto = "/home/moorthi/PycharmProjects/face_mask_server/venv/face_detector/deploy.prototxt"
model = "/home/moorthi/PycharmProjects/face_mask_server/venv/face_detector/res10_300x300_ssd_iter_140000.caffemodel"
net = cv2.dnn.readNetFromCaffe(proto, model)
vid = cv2.VideoCapture(0)

while True:

    ret, frame = vid.read()
    frame = imutils.resize(frame, width=400)

    (h, w) = frame.shape[:2]
    blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 1.0,
                                 (300, 300), (104.0, 177.0, 123.0))

    net.setInput(blob)
    detections = net.forward()

    for i in range(0, detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence < 0.5:
            continue
        box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
        (startX, startY, endX, endY) = box.astype("int")
        bv = [startX, startY, endX, endY]
        # print(type(boundingbox[0]))
        x = int(bv[0])
        y = int(bv[1])
        h = int(bv[3])
        w = int(bv[2])
        crop_img = frame[y:h, x:w]
        # crop_img = frame[startY:endX, startX:endY]

        # cv2.imwrite("car_crop.jpg", crop_img)
    cv2.imshow("Frame", crop_img)
    key = cv2.waitKey(1) & 0xFF

    if key == ord("q"):
        break

# do a bit of cleanup
vid.release()
cv2.destroyAllWindows()
