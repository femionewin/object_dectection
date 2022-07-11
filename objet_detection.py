
import cv2
import numpy as np
import sys
import os
import time

# Load Yolo
net = cv2.dnn.readNet("yolov3.weights", "yolov3.cfg")
classes = []
with open("coco.names", "r") as f:
    classes = [line.strip() for line in f.readlines()]
layer_names = net.getLayerNames()
output_layers = [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]
colors = np.random.uniform(0, 255, size=(len(classes), 3))

# Loading image

#gst_str = 'udpsrc port=14500 caps=" application/x-rtp" ! rtph264depay ! h264parse ! avdec_h264 ! appsink'
#gst_str = ('udpsrc port=14500 caps = "application/x-rtp, media=(string)video, clock-rate=(int)90000, encoding-name=(string)H264, payload=(int)96" ! rtph264depay ! decodebin ! videoconvert ! appsink')
#cap = cv2.VideoCapture(gst_str, cv2.CAP_GSTREAMER)

os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "protocol_whitelist;file,rtp,udp"
cap = cv2.VideoCapture('test.sdp', cv2.CAP_FFMPEG)

if not cap.isOpened():
        print('VideoCapture not opened')
        exit(0)

while True:

    widths =1024
    heights = 416
    dim = (widths, heights)
    channels, frame = cap.read()

    vid = cv2.resize(frame, dim, interpolation = cv2.INTER_AREA)
    height, width, channels = vid.shape

    # Detecting objects
    blob = cv2.dnn.blobFromImage(vid, 1/255, (416, 416), (0,0,0), swapRB=True, crop=False)
    net.setInput(blob)
    outs = net.forward(output_layers)

    # Showing informations on the screen
    class_ids = []
    confidences = []
    boxes = []
    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > 0.5:
                # Object detected
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)
                w = int(detection[2] * width)
                h = int(detection[3] * height)

                # Rectangle coordinates
                x = int(center_x - w / 2)
                y = int(center_y - h / 2)

                boxes.append([x, y, w, h])
                confidences.append(float(confidence))
                class_ids.append(class_id)

    indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)

    font = cv2.FONT_HERSHEY_PLAIN
    for i in range(len(boxes)):

        if i in indexes:
            x, y, w, h = boxes[i]
            label = str(classes[class_ids[i]])
            color = colors[i]
            print(color)
            cv2.rectangle(vid, (x, y), (x + w, y + h), color, 2)
            cv2.putText(vid, label, (x, y + 30), font, 3, color, 3)


    cv2.imshow("Video", vid)
    key = cv2.waitKey(1)
    if key == 27:
        break

cap.release()
cv2.destroyAllWindows()
