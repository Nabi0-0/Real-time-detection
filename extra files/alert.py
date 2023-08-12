import cv2
import numpy as np
import smtplib

# Constants
CAMERA_DISTANCE_THRESHOLD = 100  # Distance threshold in pixels for triggering the alert
ALERT_EMAIL = 'darkcloud7060@gmail.com'  # Email address for sending the alert
ALERT_PASSWORD = 'kdobahhtggtufxol'  # Password for the email account
ALERT_SUBJECT = 'Alert: Object Detected'  # Email subject for the alert

# Load the pre-trained Haar cascade XML file for object detection
objectCascade = cv2.CascadeClassifier('C:/Users/vedan/Pycharm/PycharmProjects/smart_cctv/venv/Lib/site-packages/cv2/data/haarcascade_frontalface_default.xml')

# Initialize the video capture
cap = cv2.VideoCapture(0)

# Initialize the email server
server = smtplib.SMTP('smtp.gmail.com', 587)
server.starttls()
server.login(ALERT_EMAIL, ALERT_PASSWORD)


while True:
    # Read the video frame
    ret, frame = cap.read()

    # Convert the frame to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Perform object detection
    objects = objectCascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    # Check if any object is within the specified distance threshold
    if len(objects) > 0:
        message = f'Subject: {ALERT_SUBJECT}/n/nAn object has been detected within the specified range!'
        server.sendmail(ALERT_EMAIL, ALERT_EMAIL, message)

    # Draw bounding boxes around the detected objects
    for (x, y, w, h) in objects:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 3)

    # Display the resulting frame
    cv2.imshow('Object Detection', frame)

    # Break the loop if 'q' key is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video capture, close the window, and quit the email server
cap.release()

