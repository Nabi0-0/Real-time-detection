import cv2
import mediapipe as mp
import time


class FaceDetector:
    def __init__(self, minDetectionCon=0.5):
        self.minDetectionCon = minDetectionCon

        self.mpFaceDetection = mp.solutions.face_detection
        self.faceDetection = self.mpFaceDetection.FaceDetection(
            model_selection=0,
            min_detection_confidence=self.minDetectionCon
        )

    def findFaces(self, img, draw=True):
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = self.faceDetection.process(imgRGB)

        bboxs = []

        if results.detections:
            for face_id, detection in enumerate(results.detections):
                bboxC = detection.location_data.relative_bounding_box
                ih, iw, _ = img.shape

                x = max(0, int(bboxC.xmin * iw))
                y = max(0, int(bboxC.ymin * ih))
                w = int(bboxC.width * iw)
                h = int(bboxC.height * ih)

                bbox = (x, y, w, h)
                bboxs.append([face_id, bbox, detection.score])

                if draw:
                    img = self.fancyDraw(img, bbox)
                    cv2.putText(
                        img,
                        f'{int(detection.score[0] * 100)}%',
                        (x, max(20, y - 20)),
                        cv2.FONT_HERSHEY_PLAIN,
                        2,
                        (255, 0, 255),
                        2
                    )

        return img, bboxs

    def fancyDraw(self, img, bbox, l=30, t=5, rt=1):
        x, y, w, h = bbox
        x1, y1 = x + w, y + h

        cv2.rectangle(img, (x, y), (x1, y1), (255, 0, 255), rt)

        cv2.line(img, (x, y), (x + l, y), (255, 0, 255), t)
        cv2.line(img, (x, y), (x, y + l), (255, 0, 255), t)

        cv2.line(img, (x1, y), (x1 - l, y), (255, 0, 255), t)
        cv2.line(img, (x1, y), (x1, y + l), (255, 0, 255), t)

        cv2.line(img, (x, y1), (x + l, y1), (255, 0, 255), t)
        cv2.line(img, (x, y1), (x, y1 - l), (255, 0, 255), t)

        cv2.line(img, (x1, y1), (x1 - l, y1), (255, 0, 255), t)
        cv2.line(img, (x1, y1), (x1, y1 - l), (255, 0, 255), t)

        return img

    def close(self):
        self.faceDetection.close()


def main():
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        raise RuntimeError("Unable to open webcam.")

    detector = None
    previous_time = 0

    try:
        detector = FaceDetector()

        while True:
            success, img = cap.read()

            if not success:
                print("Unable to read webcam frame.")
                break

            img, bboxs = detector.findFaces(img)

            current_time = time.time()
            delta = current_time - previous_time
            fps = 1 / delta if delta > 0 else 0
            previous_time = current_time

            cv2.putText(
                img,
                f'FPS: {int(fps)}',
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
        if detector is not None:
            detector.close()

        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
