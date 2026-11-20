import cv2
from pathlib import Path


def main():
    project_dir = Path(__file__).resolve().parent

    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        raise RuntimeError("Unable to open webcam.")

    face_cascade = cv2.CascadeClassifier(
        cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    )

    if face_cascade.empty():
        cap.release()
        raise RuntimeError("Unable to load face detection classifier.")

    img_counter = 0
    face_present = False

    try:
        while True:
            ret, frame = cap.read()

            if not ret:
                print("Unable to read webcam frame.")
                break

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            faces = face_cascade.detectMultiScale(
                gray,
                scaleFactor=1.3,
                minNeighbors=5,
                minSize=(30, 30)
            )

            if len(faces) > 0 and not face_present:
                for (x, y, w, h) in faces:
                    cv2.rectangle(
                        frame, (x, y), (x + w, y + h), (0, 255, 0), 2
                    )

                img_name = project_dir / f"opencv_frame_{img_counter}.png"
                saved = cv2.imwrite(str(img_name), frame)

                if saved:
                    print(f"{img_name.name} written!")
                    img_counter += 1
                else:
                    print("Unable to save snapshot.")

            face_present = len(faces) > 0

            cv2.imshow("Webcam", frame)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    except KeyboardInterrupt:
        print("Interrupted by user.")

    finally:
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
