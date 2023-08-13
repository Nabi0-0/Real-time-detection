##importing
#importing required libraries
import cv2
import mediapipe as mp
import time
import numpy as np
import cvzone
from cvzone.FaceMeshModule import FaceMeshDetector
import smtplib

##setup
#setting up video frame
cap= cv2.VideoCapture(0)

# Define the codec and create a VideoWriter object
fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter('output.avi', fourcc, 20.0, (640, 480))

#setting up object detection parameter
whT = 320
confThreshold = 0.9
nmsThreshold = 0.2

# Constants for alert
CAMERA_DISTANCE_THRESHOLD = 100  # Distance threshold in pixels for triggering the alert
ALERT_EMAIL = 'darkcloud7060@gmail.com'  # Email address for sending the alert
ALERT_PASSWORD = 'kdobahhtggtufxol'  # Password for the email account
ALERT_SUBJECT = 'Alert: Object Detected'  # Email subject for the alert

# Initialize the email server
server = smtplib.SMTP('smtp.gmail.com', 587) #default for gmail
server.starttls()
server.login(ALERT_EMAIL, ALERT_PASSWORD)

#initialize face mest detecter
detector = FaceMeshDetector()

#loading object detection model
modelConfiguration = "yolov3-320.cfg"
modelWeights = "yolov3-320.weights"
net = cv2.dnn.readNetFromDarknet(modelConfiguration, modelWeights)
net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)

#loading object detection classname
classesFile = "coco.names"
classNames = open(classesFile, 'rt').read().rstrip('\n').split('\n')

#object function
def findObjects(outputs, img):
    hT, wT, _ = img.shape
    bbox = []
    classIds = []
    confs = []
    for output in outputs:
        for det in output:
            scores = det[5:]
            classId = np.argmax(scores)
            confidence = scores[classId]
            if confidence > confThreshold:
                w, h = int(det[2] * wT), int(det[3] * hT)
                x, y = int((det[0] * wT) - w / 2), int((det[1] * hT) - h / 2)
                bbox.append([x, y, w, h])
                classIds.append(classId)
                confs.append(float(confidence))

    indices = cv2.dnn.NMSBoxes(bbox, confs, confThreshold, nmsThreshold)

    for i in indices:
        i = 0
        box = bbox[i]
        x, y, w, h = box
        cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 255), 2)
        cv2.putText(img, f'{classNames[classIds[i]].upper()} {int(confs[i] * 100)}%',
                    (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 255), 2)


#setting image counter
img_counter = 0

#setting time
pTime = 0

#main loop video processing
while True:
    success, img = cap.read()
    blob = cv2.dnn.blobFromImage(img, 1 / 255, (whT, whT), [0, 0, 0], 1, crop=False)
    net.setInput(blob)
    layersNames = net.getLayerNames()
    outputNames = [layersNames[i - 1] for i in net.getUnconnectedOutLayers()]
    outputs = net.forward(outputNames)
    findObjects(outputs, img)

    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = mp.solutions.face_detection.FaceDetection(0.75).process(imgRGB)

    if results.detections:
        for id, detection in enumerate(results.detections):
            bbox = detection.location_data.relative_bounding_box
            ih, iw, _ = img.shape
            bbox = int(bbox.xmin * iw), int(bbox.ymin * ih), int(bbox.width * iw), int(bbox.height * ih)
            cv2.rectangle(img, bbox, (255, 0, 255), 2)
            cv2.putText(img, f'person', (bbox[0], bbox[1] - 20), cv2.FONT_HERSHEY_PLAIN, 2, (255, 0, 255), 2)

            # Calculate face distance
            img, faces = detector.findFaceMesh(img, draw=False)
            if faces:
                face = faces[0]
                pointLeft = face[145]
                pointRight = face[374]
                w, _ = detector.findDistance(pointLeft, pointRight)
                W = 6.3  # refrence width

                # Finding distance
                f = 840 # focal length
                d = (W * f) / w
                print(d)
                cvzone.putTextRect(img, f'Depth: {int(d)}cm',
                                   (face[10][0] - 100, face[10][1] - 50),
                                   scale=2)
                #sending msg
                if len(face) > 0:
                    message = f'Subject: {ALERT_SUBJECT}/n/nAn object has been detected within the specified range!'
                    server.sendmail(ALERT_EMAIL, ALERT_EMAIL, message)

            #FPS calculation
            cTime = time.time() # current time
            fps = 1 / (cTime - pTime)
            pTime = cTime
            cv2.putText(img, f'FPS: {int(fps)}', (20, 70), cv2.FONT_HERSHEY_PLAIN,
                        3, (0, 255, 0), 2)

            #snapshot
            img_name = f"opencv_frame_{img_counter}.png"
            cv2.imwrite(img_name, img)
            print(f"{img_name} written!")
            #img_counter += 1

            #recording
           out.write(img)

    #cleaning up
    cv2.imshow("PROJECT", img)
    if cv2.waitKey(1) & 0xFF == 27: #esc hit closing
        break

cap.release()
cv2.destroyAllWindows()
