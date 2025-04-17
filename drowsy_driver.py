from scipy.spatial import distance as dist
from imutils import face_utils
from pygame import mixer
import numpy as np
import imutils
import dlib
import cv2
#eye aspect ratio(EAR) = sum of vertical eye distances / 2*sum of horizontal eye distances

def eye_aspect_ratio(eye):
    # Compute the euclidean distances between the two sets of vertical eye landmarks
    A = dist.euclidean(eye[1], eye[5])
    B = dist.euclidean(eye[2], eye[4])
    # Compute the euclidean distance between the horizontal eye landmark
    C = dist.euclidean(eye[0], eye[3])
    # Compute the eye aspect ratio
    ear = (A + B) / (2.0 * C)
    return ear
flag = 0 # Flag to indicate if the alarm is playing
frame_check = 20 # Number of frames to check for drowsiness
# Load the facial landmark predictor model and define the eye landmarks 

(lStart,lEnd) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"] # Indices for the left eye landmarks
(rStart,rEnd) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"] # Indices for the right eye landmarks

detect = dlib.get_frontal_face_detector() # Initialize the face detector
predict = dlib.shape_predictor("./shape_predictor_68_face_landmarks.dat") # Load the facial landmarks predictor

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read() # Read a frame from the camera
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) # Convert the frame to grayscale
    subjects = detect(gray, 0) # Detect faces in the grayscale frame
    for subject in subjects:
        shape = predict(gray, subject) # Predict the facial landmarks
        shape = face_utils.shape_to_np(shape) # Convert the landmarks to a NumPy array
        leftEye = shape[lStart:lEnd]
        rightEye = shape[rStart:rEnd]
        leftEar = eye_aspect_ratio(leftEye)
        rightEar = eye_aspect_ratio(rightEye)
        ear = (leftEar + rightEar) / 2.0
        # Draw the eye landmarks on the frame
        leftEyeHull = cv2.convexHull(leftEye)
        rightEyeHull = cv2.convexHull(rightEye)
        cv2.drawContours(frame, [leftEyeHull], -1, (0, 255, 0), 1)
        cv2.drawContours(frame, [rightEyeHull], -1, (0, 255, 0), 1)

        if ear < 0.25:
            flag += 1
            # If the eye aspect ratio is below a certain threshold for a number of frames, trigger the alarm
            print(flag)
            if flag >= frame_check:
                cv2.putText(frame, "DROWSINESS ALERT!", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                # If the alarm is not already playing, play the alarm sound
                mixer.init()
                mixer.music.load("alarm.wav")
                mixer.music.play(0)
        else:
            flag = 0 # Reset the flag if the eye aspect ratio is above the threshold

    cv2.imshow("Frame", frame) # Display the frame in a window
    cv2.waitKey(1) # Wait for 1 millisecond to allow the window to update

