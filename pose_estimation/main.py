import cv2
import time
import mediapipe as mp
import json

from body_info.keypoints import detectPose
from utils import calculateAngle
    

# Initializing mediapipe pose class.
mp_pose = mp.solutions.pose

# Setup Pose function for video.
pose_video = mp_pose.Pose(static_image_mode=False, min_detection_confidence=0.5, model_complexity=1)

# Initialize the VideoCapture object to read from the webcam.
video = cv2.VideoCapture(0)
video.set(3,1280)
video.set(4,960)

# Initialize a resizable window.
cv2.namedWindow('Frames', cv2.WINDOW_NORMAL)
count_frame = 0
replay = 0
last_time = time.time()

"""
INITIALIZE SHOULDER 3D COORDINATES
"""
with open('./database/shoulder.json', 'w') as f1:
    json.dump({
        "SHOULDER_0": [0, 0, 0]
    }, f1, indent=4)

"""
INITIALIZE HIP 3D COORDINATES
"""
with open('./database/hip.json', 'w') as f2:
    json.dump({
        "HIP_0": [0, 0, 0]
    }, f2, indent=4)


"""
INITIALIZE KNEE 3D COORDINATES
"""
with open('./database/knee.json', 'w') as f3:
    json.dump({
        "KNEE_0": [0, 0, 0]
    }, f3, indent=4)
while True:
    count_frame += 1
    if video.get(cv2.CAP_PROP_FRAME_COUNT) == count_frame:
        video.set(cv2.CAP_PROP_POS_FRAMES, 0)
        count_frame = 1
        replay = 1

    ret, frame = video.read()
    duration = time.time() - last_time

    last_time = time.time()
    fps = str(round((1/duration), 2))

    if ret:
        #frame = imutils.resize(frame, width=800, inter=cv2.INTER_LINEAR)
        cv2.putText(frame, "fps: " + fps, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 1, cv2.LINE_AA)

        # # Flip the frame horizontally for natural (selfie-view) visualization.
        # frame = cv2.flip(frame, 1)
        
        # Get the width and height of the frame
        frame_height, frame_width, _ =  frame.shape
        
        # Resize the frame while keeping the aspect ratio.
        frame = cv2.resize(frame, (int(frame_width * (640 / frame_height)), 640))
        
        # Perform Pose landmark detection.
        frame, landmarks, real_landmarks = detectPose(count_frame, frame, pose_video, display=False)
        
        try:
            if landmarks:
                
                # # Perform the Pose Classification.
                # frame, _ = classifyPose(landmarks, frame, display=False)
                with open('./database/shoulder.json') as f1:
                    config1 = json.load(f1)

                shoulder = config1[f'SHOULDER_{count_frame}']

                with open('./database/hip.json') as f2:
                    config2 = json.load(f2)

                hip = config2[f'HIP_{count_frame}']

                with open('./database/knee.json') as f3:
                    config3 = json.load(f3)

                knee = config3[f'KNEE_{count_frame}']

                low_back_angle = round(abs(calculateAngle(shoulder, hip, knee) - 180), 3)

                cv2.putText(frame, f"Spine Angle: {low_back_angle}", (50, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 1, cv2.LINE_AA)

            cv2.imshow("Frames", frame)

        except TypeError as e:
            print(e)
            break

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break