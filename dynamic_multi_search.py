# use python3 dynamic_multi_search.py --add-person

import cv2
import face_recognition
import pandas as pd
from ultralytics import YOLO
from tracker import Tracker
import datetime
import os
import argparse


# Load the known images for face recognition
known_images = [
    face_recognition.load_image_file("./vinayak.jpg"),
    face_recognition.load_image_file("../faces/rdj.jpeg"),
    # Add more known images here for each person you want to recognize
]

known_faces = [face_recognition.face_encodings(img)[0] for img in known_images]
# Input the names of the persons to recognize
known_names = ["vinayak","rdj"]  # Add names for each known person

# Function to add a new person to the known faces list
def add_new_person(known_faces, known_names):
    person_name = input("Enter the name of the new person: ")
    image_path = input("Enter the path to the image of the new person: ")

    if not os.path.isfile(image_path):
        print("Invalid image path. Please try again.")
        return known_faces, known_names

    new_image = face_recognition.load_image_file(image_path)
    new_face_encoding = face_recognition.face_encodings(new_image)

    if not new_face_encoding:
        print("No face detected in the provided image. Please try again.")
        return known_faces, known_names

    known_names.append(person_name)
    known_faces.append(new_face_encoding[0])
    print(f"New person '{person_name}' added successfully!")

    return known_faces, known_names


# Initialize YOLO model for object detection
model = YOLO('yolov8s.pt')

# Number of webcams to use
num_webcams = 1

# Create a list of cv2.VideoCapture objects for each webcam
caps = [cv2.VideoCapture(i) for i in range(num_webcams)]

with open("./coco.names", "r") as my_file:
    class_list = my_file.read().split("\n")

# Create an argument parser for CLI
parser = argparse.ArgumentParser(description="Real-time person recognition")
parser.add_argument("--add-person", action="store_true", help="Add a new person for recognition")
args = parser.parse_args()


count = 0
tracker = Tracker()
people_count = 0  # Counter for people in the current frame
max_people_count = 0  # Maximum people count so far

first_detection_times = [None] * num_webcams
last_detection_times = [None] * num_webcams

while True:
    for i in range(num_webcams):
        ret, frame = caps[i].read()
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
        recognized_names = []

        for (top, right, bottom, left), face_encoding in zip(detected_faces, face_recognition.face_encodings(frame)):
            matched_name = "Unknown"
            detected_cam = None  # Variable to store the camera ID


            if args.add_person:
                known_faces, known_names = add_new_person(known_faces, known_names)
                args.add_person = False  # Reset the flag
                # continue  # Skip face recognition for this frame

            else:
                # Match with the known faces
                for id, known_face in enumerate(known_faces):
                    result = face_recognition.compare_faces([known_face], face_encoding)
                    if result[0]:
                        matched_name = known_names[id]
                        detected_cam = i  # Store the camera ID
                        break

            if matched_name != "Unknown":
                frame = cv2.putText(frame, f'Person: {matched_name}', (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
                print(f'{matched_name} Detected on Camera {detected_cam}')
                # Show the detected frame alongside the main feed
                detected_frame = frame[top:bottom, left:right]

                if first_detection_times[i] is None:
                    first_detection_times[i] = datetime.datetime.now()

                last_detection_times[i] = datetime.datetime.now()

                detection_info = f"Webcam ID: {i} | First Detected: {first_detection_times[i]} | Last Detected: {last_detection_times[i]}"
                # frame_copy = frame.copy()
                # frame_copy[0:detected_frame.shape[0], 0:detected_frame.shape[1]] = detected_frame
                # frame_copy = cv2.putText(detection_info, (10, frame_copy.shape[0] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
                cv2.imshow(f"Detected {matched_name} (Cam {i})", frame)


        bbox_idx = tracker.update(detected_faces)
        for bbox in bbox_idx:
            x3, y3, x4, y4, id = bbox
            cv2.rectangle(frame, (x3, y3), (x4, y4), (0, 255, 0), 2)
            cv2.putText(frame, str(id), (x3, y3), cv2.FONT_HERSHEY_COMPLEX, 0.5, (255, 0, 0), 1)

        # Update the maximum people count so far for each webcam
        max_people_count = max(people_count, max_people_count)

        cv2.putText(frame, f"Total People (Cam {i} - Current): {people_count}", (10, 30), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 255), 2)
        cv2.putText(frame, f"Max People Count (Cam {i}): {max_people_count}", (10, 70), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 255), 2)
        cv2.imshow(f"RGB (Cam {i})", frame)
        if cv2.waitKey(1) & 0xFF == 27:
            break

    if cv2.waitKey(1) & 0xFF == 27:
        break

# Release all the webcam objects
for cap in caps:
    cap.release()

cv2.destroyAllWindows()