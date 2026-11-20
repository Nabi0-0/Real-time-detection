import cv2
import mediapipe as mp
import time



def facemask():
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

    if not cap.isOpened():
        raise RuntimeError("Unable to open webcam.")

    mp_face_detection = mp.solutions.face_detection
    face_detection = mp_face_detection.FaceDetection(
        model_selection=0,
        min_detection_confidence=0.75
    )

    previous_time = 0

    try:
        while True:
            success, img = cap.read()

            if not success:
                print("Unable to read webcam frame.")
                break

            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            results = face_detection.process(img_rgb)

            if results.detections:
                for detection in results.detections:
                    bbox_c = detection.location_data.relative_bounding_box
                    ih, iw, _ = img.shape

                    bbox = (
                        int(bbox_c.xmin * iw),
                        int(bbox_c.ymin * ih),
                        int(bbox_c.width * iw),
                        int(bbox_c.height * ih)
                    )

                    cv2.rectangle(
                        img,
                        bbox,
                        (255, 0, 255),
                        2
                    )

                    cv2.putText(
                        img,
                        f"{int(detection.score[0] * 100)}%",
                        (bbox[0], bbox[1] - 20),
                        cv2.FONT_HERSHEY_PLAIN,
                        2,
                        (255, 0, 255),
                        2
                    )

            current_time = time.time()
            delta = current_time - previous_time
            fps = 1 / delta if delta > 0 else 0
            previous_time = current_time

            cv2.putText(
                img,
                f"FPS: {int(fps)}",
                (20, 70),
                cv2.FONT_HERSHEY_PLAIN,
                3,
                (0, 255, 0),
                2
            )

            cv2.imshow("Image", img)

            if cv2.waitKey(1) & 0xFF == 27:
                break

    except KeyboardInterrupt:
        print("Interrupted by user.")

    finally:
        cap.release()
        face_detection.close()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    facemask()
