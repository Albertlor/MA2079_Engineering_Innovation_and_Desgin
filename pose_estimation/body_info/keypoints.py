import cv2
import mediapipe as mp
import matplotlib.pyplot as plt
import json

from utils import midpoint


# Initializing mediapipe pose class.
mp_pose = mp.solutions.pose

# Setting up the Pose function.
pose = mp_pose.Pose(static_image_mode=True, min_detection_confidence=0.7, model_complexity=2)

# Initializing mediapipe drawing class, useful for annotation.
mp_drawing = mp.solutions.drawing_utils

LEFT_SHOULDER = mp_pose.PoseLandmark.LEFT_SHOULDER.value
RIGHT_SHOULDER = mp_pose.PoseLandmark.RIGHT_SHOULDER.value
LEFT_HIP = mp_pose.PoseLandmark.LEFT_HIP.value
RIGHT_HIP = mp_pose.PoseLandmark.RIGHT_HIP.value
LEFT_KNEE = mp_pose.PoseLandmark.LEFT_KNEE.value
RIGHT_KNEE = mp_pose.PoseLandmark.RIGHT_KNEE.value

def detectPose(frame, image, pose, display=True):
    '''
    This function performs pose detection on an image.
    Args:
        image: The input image with a prominent person whose pose landmarks needs to be detected.
        pose: The pose setup function required to perform the pose detection.
        display: A boolean value that is if set to true the function displays the original input image, the resultant image, 
                 and the pose landmarks in 3D plot and returns nothing.
    Returns:
        output_image: The input image with the detected pose landmarks drawn.
        landmarks: A list of detected landmarks converted into their original scale.
    '''
    
    # Create a copy of the input image.
    output_image = image.copy()
    
    # Convert the image from BGR into RGB format.
    imageRGB = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Perform the Pose Detection.
    results = pose.process(imageRGB)
    
    # Retrieve the height and width of the input image.
    height, width, _ = image.shape
    
    # Initialize a list to store the detected landmarks.
    landmarks = []
    
    # Check if any landmarks are detected.
    if results.pose_landmarks:
    
        # Draw Pose landmarks on the output image.
        mp_drawing.draw_landmarks(image=output_image, landmark_list=results.pose_landmarks,
                                  connections=mp_pose.POSE_CONNECTIONS)
        
        confidence_list = []
        for lm in results.pose_landmarks.landmark:
            if lm.visibility >= 0.7:
                confidence_list.append(lm.visibility)

        print(f'Confidence List: {confidence_list}')
        
        if len(confidence_list) == 33:
            # Iterate over the detected landmarks.
            for landmark in results.pose_landmarks.landmark:
                # Append the landmark into the list.
                landmarks.append((int(landmark.x * width), int(landmark.y * height),
                                    (landmark.z * width)))
                
            shoulder = midpoint(landmarks[LEFT_SHOULDER], landmarks[RIGHT_SHOULDER])
            hip = midpoint(landmarks[LEFT_HIP], landmarks[RIGHT_HIP])
            knee = midpoint(landmarks[LEFT_KNEE], landmarks[RIGHT_KNEE])
            
            """
            SHOULDER 3D COORDINATES
            """
            with open('./database/shoulder.json') as f1:
                config1 = json.load(f1)

            config1[f'SHOULDER_{frame}'] = shoulder

            with open('./database/shoulder.json', 'w') as f1:
                json.dump(config1, f1, indent=4)


            """
            HIP 3D COORDINATES
            """
            with open('./database/hip.json') as f2:
                config2 = json.load(f2)

            config2[f'HIP_{frame}'] = hip

            with open('./database/hip.json', 'w') as f2:
                json.dump(config2, f2, indent=4)


            """
            KNEE 3D COORDINATES
            """
            with open('./database/knee.json') as f3:
                config3 = json.load(f3)

            config3[f'KNEE_{frame}'] = knee

            with open('./database/knee.json', 'w') as f3:
                json.dump(config3, f3, indent=4)
    
    # Check if the original input image and the resultant image are specified to be displayed.
    if display:
    
        # Display the original input image and the resultant image.
        plt.figure(figsize=[22,22])
        plt.subplot(121);plt.imshow(image[:,:,::-1]);plt.title("Original Image");plt.axis('off');
        plt.subplot(122);plt.imshow(output_image[:,:,::-1]);plt.title("Output Image");plt.axis('off');
        
        # Also Plot the Pose landmarks in 3D.
        mp_drawing.plot_landmarks(results.pose_world_landmarks, mp_pose.POSE_CONNECTIONS)
        
    # Otherwise
    else:
        
        # Return the output image and the found landmarks.
        return output_image, landmarks, results.pose_world_landmarks