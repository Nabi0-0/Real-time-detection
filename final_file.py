##importing
#importing required libraries
import cv2
import mediapipe as mp
import time
import numpy as np
from pathlib import Path

project_dir = Path(__file__).resolve().parent

##setup
cap = None
out = None
face_mesh = None
face_detector = None

try:
    # Initialize webcam
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        raise RuntimeError("Unable to open webcam.")

    # Get actual webcam resolution
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    if frame_width <= 0 or frame_height <= 0:
        raise RuntimeError("Unable to determine webcam resolution.")

    # Initialize video recording
    fourcc = cv2.VideoWriter_fourcc(*"XVID")
    out = cv2.VideoWriter(
        str(project_dir / "output.avi"),
        fourcc,
        20.0,
        (frame_width, frame_height)
    )

    if not out.isOpened():
        raise RuntimeError("Unable to initialize video recording.")

    #setting up object detection parameter
    whT = 608
    confThreshold = 0.9
    nmsThreshold = 0.2

    #initialize face mest detecter
    face_mesh = mp.solutions.face_mesh.FaceMesh(
        static_image_mode=False,
        max_num_faces=2,
        refine_landmarks=False,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    )

    #loading object detection model
    object_detection_dir = project_dir / "Object_detection"
    modelConfiguration = object_detection_dir / "yolov3-320.cfg"
    modelWeights = object_detection_dir / "yolov3.weights"
    net = cv2.dnn.readNetFromDarknet(modelConfiguration, modelWeights)
    net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
    net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)

    # Cache YOLO output layer names (they do not change between frames)
    layersNames = net.getLayerNames()
    outputNames = [
        layersNames[i - 1] for i in net.getUnconnectedOutLayers()
    ]

    #loading object detection classname
    classesFile = object_detection_dir / "coco.names"
    with open(classesFile, "r", encoding="utf-8") as file:
        classNames = file.read().rstrip("\n").split("\n")

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

        if len(indices) > 0:
            for i in np.array(indices).flatten():
                box = bbox[i]
                x, y, w, h = box
                cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 255), 2)
                cv2.putText(img, f'{classNames[classIds[i]].upper()} {int(confs[i] * 100)}%',
                            (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 255), 2)


    #setting image counter
    img_counter = 0

    #setting time
    pTime = 0

    # Initialize face detection once
    face_detector = mp.solutions.face_detection.FaceDetection(
        model_selection=0,
        min_detection_confidence=0.75
    )

    # Main loop video processing
    while True:
        success, img = cap.read()

        if not success:
            print("Unable to read webcam frame.")
            break

        ih, iw, _ = img.shape

        # YOLO object detection
        blob = cv2.dnn.blobFromImage(
            img, 1 / 255, (whT, whT), [0, 0, 0], 1, crop=False
        )
        net.setInput(blob)


        outputs = net.forward(outputNames)
        findObjects(outputs, img)

        # Convert frame to RGB for MediaPipe
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # Face detection
        results = face_detector.process(imgRGB)

        if results.detections:
            for detection in results.detections:
                bbox = detection.location_data.relative_bounding_box

                x = int(bbox.xmin * iw)
                y = int(bbox.ymin * ih)
                width = int(bbox.width * iw)
                height = int(bbox.height * ih)

                cv2.rectangle(
                    img, (x, y), (x + width, y + height),
                    (255, 0, 255), 2
                )

                cv2.putText(
                    img, "person", (x, y - 20),
                    cv2.FONT_HERSHEY_PLAIN, 2,
                    (255, 0, 255), 2
                )

        # Face mesh and distance estimation
        mesh_results = face_mesh.process(imgRGB)

        if mesh_results.multi_face_landmarks:
            face = mesh_results.multi_face_landmarks[0].landmark

            point_left = (
                int(face[145].x * iw),
                int(face[145].y * ih)
            )

            point_right = (
                int(face[374].x * iw),
                int(face[374].y * ih)
            )

            w = np.linalg.norm(
                np.array(point_left) - np.array(point_right)
            )

            if w > 0:
                W = 6.3
                f = 840
                distance = (W * f) / w

                print(f"Estimated distance: {distance:.2f} cm")

                text_position = (
                    int(face[10].x * iw) - 100,
                    int(face[10].y * ih) - 50
                )

                cv2.putText(
                    img,
                    f"Depth: {int(distance)}cm",
                    text_position,
                    cv2.FONT_HERSHEY_PLAIN,
                    2,
                    (255, 255, 255),
                    2
                )

        # Email alerts temporarily disabled pending proper trigger logic.

        # FPS calculation
        cTime = time.time()
        fps = 1 / (cTime - pTime) if cTime != pTime else 0
        pTime = cTime

        cv2.putText(
            img, f"FPS: {int(fps)}", (20, 70),
            cv2.FONT_HERSHEY_PLAIN, 3, (0, 255, 0), 2
        )

        # Snapshot: press S to save the current frame
        key = cv2.waitKey(1) & 0xFF

        if key == ord("s"):
            img_name = project_dir / f"opencv_frame_{img_counter}.png"
            saved = cv2.imwrite(img_name, img)

            if saved:
                print(f"Snapshot saved: {img_name}")
                img_counter += 1
            else:
                print("Unable to save snapshot.")

        # Recording
        out.write(img)

        # Display
        cv2.imshow("PROJECT", img)

        if key == 27:
            break

finally:
    if out is not None:
        out.release()
    if face_detector is not None:
        face_detector.close()
    if cap is not None:
        cap.release()
    if face_mesh is not None:
        face_mesh.close()
    cv2.destroyAllWindows()
