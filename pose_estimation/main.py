# https://www.youtube.com/watch?v=aySurynUNAw&t=1930s&ab_channel=BleedAIAcademy

import cv2
import time
import json
import numpy as np

from utils import magnitude
from body_info.keypoints import Pose
    

# Initialize the VideoCapture object to read from the webcam.
video = cv2.VideoCapture(0)
video.set(3,1280)
video.set(4,960)

# Initialize a resizable window.
cv2.namedWindow('Frames', cv2.WINDOW_NORMAL)
count_frame = 0
count = 0
g_dir = 0
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
        
        # Get the width and height of the frame
        frame_height, frame_width, _ =  frame.shape
        
        # Resize the frame while keeping the aspect ratio.
        frame = cv2.resize(frame, (int(frame_width * (640 / frame_height)), 640))
        
        pose = Pose(count_frame, frame, display=False)
        
        # Perform Pose landmark detection.
        frame, landmarks, real_landmarks, confidence_index, front = pose.detectPose()

        if front:
            cv2.putText(frame, "Person is facing towards the robot, do a 90deg turn", (50, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 1, cv2.LINE_AA)
        else:
            cv2.putText(frame, "Stay in this manner", (50, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 1, cv2.LINE_AA)
        
        try:
            if front == None:
                cv2.putText(frame, f"Region of Interest is not detected", (50, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 1, cv2.LINE_AA)
            if front == 0:
                if confidence_index==2:
                    
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

                    if count == 0:
                        r = np.array(shoulder) - np.array(hip)
                        g_dir = np.multiply(r, 1/(magnitude(r)))
                        count += 1

                    low_back_angle = round(abs(pose.calculateAngle(g_dir, shoulder, hip, knee)), 3)

                    cv2.putText(frame, f"Spine Angle: {low_back_angle}", (50, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 1, cv2.LINE_AA)

                else:
                    cv2.putText(frame, f"Region of Interest is not detected", (50, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 1, cv2.LINE_AA)

            cv2.imshow("Frames", frame)

        except TypeError as e:
            print(e)
            pass

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break