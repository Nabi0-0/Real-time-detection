import cv2

# Open the webcam
cap = cv2.VideoCapture(0)

# Define the codec and create a VideoWriter object
fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter('output.avi', fourcc, 20.0, (640, 480))

while cap.isOpened():
    ret, frame = cap.read()

    if ret:
        # Display the frame
        cv2.imshow('Webcam', frame)

        # Write the frame to the video file
        out.write(frame)

        # Press 'q' to exit
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    else:
        break

# Release the webcam and the video writer
cap.release()
out.release()

# Close all OpenCV windows
cv2.destroyAllWindows()
