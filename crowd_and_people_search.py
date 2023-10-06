import cv2
import face_recognition
import pandas as pd
from ultralytics import YOLO
from tracker import Tracker
import datetime

# Load the known image for face recognition
known_image = face_recognition.load_image_file("./vinayak.jpg")
known_faces = face_recognition.face_encodings(known_image)

# Input the name of the person to recognize
person_name = "vinayak"

# Load YOLO model for object detection
model = YOLO('yolov8s.pt')

def RGB(event, x, y, flags, param):
    if event == cv2.EVENT_MOUSEMOVE:
        colorsBGR = [x, y]
        print(colorsBGR)

cv2.namedWindow('RGB')
cv2.setMouseCallback('RGB', RGB)

cap = cv2.VideoCapture("https://10.7.0.226:8080/video")

with open("./coco.names", "r") as my_file:
    class_list = my_file.read().split("\n")

count = 0
tracker = Tracker()
people_count = 0  # Counter for people in the current frame
max_people_count = 0  # Maximum people count so far
first_detection_time = None
last_detection_time = None

while True:
    ret, frame = cap.read()
    if not ret:
        break
    count += 1
    if count % 3 != 0:
        continue

    frame = cv2.resize(frame, (1020, 500))

    # Perform object detection with YOLO
    results = model.predict(frame)
    a = results[0].boxes.data
    px = pd.DataFrame(a).astype("int")

    people_count = 0  # Reset the counter for people in the current frame

    detected_faces = []

    for _, (x1, y1, x2, y2, _, d) in px.iterrows():
        c = class_list[d]
        if c == 'person':  # Check if the detected object is a person
            people_count += 1
        detected_faces.append([x1, y1, x2, y2])

    # Face recognition for each detected face
    for (top, right, bottom, left), face_encoding in zip(detected_faces, face_recognition.face_encodings(frame)):
        # Match with the known faces
        results = face_recognition.compare_faces(known_faces, face_encoding)

        if any(results):
            # Known face found
            frame = cv2.putText(frame, f'Person: {person_name} found', (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
            print(f'{person_name} Detected')

            # Show the detected frame alongside the main feed
            detected_frame = frame[top:bottom, left:right]

            if first_detection_time is None:
                first_detection_time = datetime.datetime.now()

            last_detection_time = datetime.datetime.now()

            detection_info = f"Webcam ID: 0 | First Detected: {first_detection_time} | Last Detected: {last_detection_time}"
            frame_copy = frame.copy()
            frame_copy[0:detected_frame.shape[0], 0:detected_frame.shape[1]] = detected_frame
            frame_copy = cv2.putText(frame_copy, detection_info, (10, frame_copy.shape[0] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
            cv2.imshow("Detected Frame", frame_copy)

    bbox_idx = tracker.update(detected_faces)
    for bbox in bbox_idx:
        x3, y3, x4, y4, id = bbox
        cx = (x3 + x4) // 2
        cy = (y3 + y4) // 2
        cv2.rectangle(frame, (x3, y3), (x4, y4), (0, 255, 0), 2)
        cv2.putText(frame, str(id), (x3, y3), cv2.FONT_HERSHEY_COMPLEX, 0.5, (255, 0, 0), 1)

    # Update the maximum people count so far
    max_people_count = max(people_count, max_people_count)

    # Display the total number of people in the current frame and the maximum so far
    cv2.putText(frame, f"Total People (Current): {people_count}", (10, 30), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 255), 2)
    cv2.putText(frame, f"Max People Count: {max_people_count}", (10, 70), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 255), 2)

    cv2.imshow("RGB", frame)
    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
