import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model

from pytube import YouTube


model = load_model('../models/fire_classifier.h5')


# # Initialize the YouTube object
video_url1 = 'https://www.youtube.com/watch?v=tJmAgfnTii8'

yt1 = YouTube(video_url1)

# Choose the stream with the highest resolution (or any other preferred stream)
stream1 = yt1.streams.filter(file_extension='mp4').get_highest_resolution()

# Open the video stream from YouTube
cap1 = cv2.VideoCapture(stream1.url)

# cap = cv2.VideoCapture(0)  # Use the default camera (usually the built-in webcam)

def detect_fire(frame):
    # Resize the frame to match the input shape expected by the model
    frame = cv2.resize(frame, (256, 256))
    frame = frame / 255.0  # Normalize pixel values to the range [0, 1]
    frame = np.expand_dims(frame, axis=0)  # Add batch dimension

    # Perform inference with the model
    prediction = model.predict(frame)[0]

    # Check if the prediction indicates fire
    if prediction[0] > 0.5:
        return True
    else:
        return False

while True:
    # Capture a frame from the webcam
    ret1, frame1 = cap1.read()

    # Perform fire detection on the captured frame
    is_fire1 = detect_fire(frame1)

    # Display the result on the frame
    if is_fire1:
        cv2.putText(frame1, "Fire Detected!", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    # Display the webcam feed with detection result
    cv2.imshow("Fire Detection", frame1)

    # Exit the loop when 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the webcam and close all OpenCV windows
cap1.release()
cv2.destroyAllWindows()
















# import cv2
# import numpy as np
# from tensorflow.keras.models import load_model
# from pytube import YouTube

# # Load the fire detection model
# model = load_model('../models/fire_classifier.h5')

# # Initialize YouTube objects and video streams for each URL
# video_urls = [
#     'https://www.youtube.com/watch?v=tJmAgfnTii8',
#     'https://www.youtube.com/watch?v=tJmAgfnTii8',
#     'https://www.youtube.com/watch?v=oXT6NNrpFg8'
# ]

# yt_streams = []
# caps = []

# for video_url in video_urls:
#     try:
#         yt = YouTube(video_url)
#         stream = yt.streams.filter(file_extension='mp4').get_highest_resolution()
#         yt_streams.append(yt)
#         caps.append(cv2.VideoCapture(stream.url))
#         print(f"Successfully opened video capture for YouTube link: {video_url}")
#     except Exception as e:
#         print(f"Error opening video capture for YouTube link {video_url}: {e}")

# # Create windows for each video feed
# cv2.namedWindow("Video Feed 1", cv2.WINDOW_NORMAL)
# cv2.namedWindow("Video Feed 2", cv2.WINDOW_NORMAL)
# cv2.namedWindow("Video Feed 3", cv2.WINDOW_NORMAL)

# def detect_fire(frame):
#     # Resize the frame to match the input shape expected by the model
#     frame = cv2.resize(frame, (256, 256))
#     frame = frame / 255.0  # Normalize pixel values to the range [0, 1]
#     frame = np.expand_dims(frame, axis=0)  # Add batch dimension

#     # Perform inference with the model
#     prediction = model.predict(frame)[0]

#     # Check if the prediction indicates fire
#     if prediction[0] > 0.5:
#         return True
#     else:
#         return False

# while True:
#     for i, cap in enumerate(caps):
#         ret, frame = cap.read()
#         if not ret:
#             break

#         # Perform fire detection on the captured frame
#         is_fire = detect_fire(frame)

#         # Display the result on the frame
#         if is_fire:
#             cv2.putText(frame, "Fire Detected!", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

#         # Display the video feed with detection result in separate windows
#         if i == 0:
#             cv2.imshow("Video Feed 1", frame)
#         elif i == 1:
#             cv2.imshow("Video Feed 2", frame)
#         elif i == 2:
#             cv2.imshow("Video Feed 3", frame)

#     # Exit the loop when 'q' is pressed
#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         break

# # Release the video captures and close all OpenCV windows
# for cap in caps:
#     cap.release()
# cv2.destroyAllWindows()
