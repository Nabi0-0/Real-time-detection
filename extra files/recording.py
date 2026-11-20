import cv2


def main():
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        raise RuntimeError("Unable to open webcam.")

    out = None

    try:
        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        if frame_width <= 0 or frame_height <= 0:
            raise RuntimeError("Unable to determine webcam resolution.")

        fourcc = cv2.VideoWriter_fourcc(*"XVID")
        out = cv2.VideoWriter(
            "output.avi",
            fourcc,
            20.0,
            (frame_width, frame_height)
        )

        if not out.isOpened():
            raise RuntimeError("Unable to initialize video recording.")

        while True:
            ret, frame = cap.read()

            if not ret:
                print("Unable to read webcam frame.")
                break

            cv2.imshow("Webcam", frame)
            out.write(frame)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    except KeyboardInterrupt:
        print("Interrupted by user.")

    finally:
        if out is not None:
            out.release()

        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
