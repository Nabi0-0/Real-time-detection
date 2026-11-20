from email.message import EmailMessage
import os
import smtplib
import time

import cv2


ALERT_SUBJECT = "Alert: Face Detected"
ALERT_COOLDOWN = 60


def main():
    alert_email = os.getenv("ALERT_EMAIL")
    alert_password = os.getenv("ALERT_PASSWORD")

    cascade_path = (
        cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    )
    face_cascade = cv2.CascadeClassifier(cascade_path)

    if face_cascade.empty():
        raise RuntimeError("Unable to load Haar Cascade classifier.")

    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        raise RuntimeError("Unable to open webcam.")

    server = None

    try:
        if alert_email and alert_password:
            try:
                server = smtplib.SMTP("smtp.gmail.com", 587, timeout=10)
                server.starttls()
                server.login(alert_email, alert_password)
                print("Email alerts enabled.")
            except (smtplib.SMTPException, OSError) as error:
                print(f"Email alerts unavailable: {error}")
                if server is not None:
                    try:
                        server.quit()
                    except (smtplib.SMTPException, OSError):
                        pass
                    server = None
        else:
            print(
                "Email alerts disabled. Configure ALERT_EMAIL "
                "and ALERT_PASSWORD."
            )

        last_alert_time = 0

        while True:
            success, frame = cap.read()

            if not success:
                print("Unable to read webcam frame.")
                break

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            faces = face_cascade.detectMultiScale(
                gray,
                scaleFactor=1.1,
                minNeighbors=5,
                minSize=(30, 30)
            )

            current_time = time.monotonic()

            if len(faces) > 0 and server is not None:
                if current_time - last_alert_time >= ALERT_COOLDOWN:
                    message = EmailMessage()
                    message["Subject"] = ALERT_SUBJECT
                    message["From"] = alert_email
                    message["To"] = alert_email
                    message.set_content(
                        "A face has been detected by the webcam."
                    )

                    try:
                        server.send_message(message)
                        print("Face detection alert sent.")
                        last_alert_time = current_time
                    except (smtplib.SMTPException, OSError) as error:
                        print(f"Unable to send alert: {error}")

            for (x, y, w, h) in faces:
                cv2.rectangle(
                    frame, (x, y), (x + w, y + h), (0, 255, 0), 3
                )

            cv2.imshow("Face Detection Alert", frame)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    except KeyboardInterrupt:
        print("Interrupted by user.")

    finally:
        cap.release()

        if server is not None:
            try:
                server.quit()
            except (smtplib.SMTPException, OSError):
                pass

        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
