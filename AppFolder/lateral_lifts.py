import cv2
import mediapipe as mp
import numpy as np
import time
from flask import Flask, render_template, Response  # Added Response import
app = Flask(__name__)


# Initialize Mediapipe Pose Model
mp_pose = mp.solutions.pose
pose = mp_pose.Pose()


# Calculate the angle between 3 Points
def calc_angle(a, b, c):  # 3D points
    a = np.array([a.x, a.y])  # Reduce 3D point to 2D
    b = np.array([b.x, b.y])  # Reduce 3D point to 2D
    c = np.array([c.x, c.y])  # Reduce 3D point to 2D

    ab = np.subtract(a, b)
    bc = np.subtract(b, c)

    theta = np.arccos(np.dot(ab, bc) / (np.linalg.norm(ab) * np.linalg.norm(bc)))  # A.B = |A||B|cos(x) where x is the angle b/w A and B
    theta = 180 - 180 * theta / 3.14  # Convert radians to degrees
    return np.round(theta, 2)

window_width = 900
window_height = 700

def infer():
    mp_drawing = mp.solutions.drawing_utils  # Connecting Keypoints Visuals
    mp_pose = mp.solutions.pose  # Keypoint detection model
    left_flag = None  # Flag which stores hand position(Either UP or DOWN)
    left_count = 0  # Storage for count of bicep curls
    right_flag = None
    right_count = 0
    line_color = (0, 0, 255)  # Initial color is red
    prev_left_count = 0
    prev_right_count = 0
    green_time_start = 0
    green_duration = 1  # Duration in seconds for line color to stay green

    cap = cv2.VideoCapture(0)
    pose = mp_pose.Pose(min_detection_confidence=0.7, min_tracking_confidence=0.5)  # Lnadmark detection model instance
    while cap.isOpened():
        _, frame = cap.read()
        frame = cv2.flip(frame, 1)
        resized_frame = cv2.resize(frame, (window_width, window_height))

        # BGR to RGB
        image = cv2.cvtColor(resized_frame, cv2.COLOR_BGR2RGB)  # Convert BGR frame to RGB
        image.flags.writeable = False

        # Make Detections
        results = pose.process(image)  # Get landmarks of the object in frame from the model

        # Back to BGR
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)  # Convert RGB back to BGR

        try:
            # Extract Landmarks
            landmarks = results.pose_landmarks.landmark
            left_shoulder = landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER]
            left_elbow = landmarks[mp_pose.PoseLandmark.LEFT_ELBOW]
            left_wrist = landmarks[mp_pose.PoseLandmark.LEFT_WRIST]
            right_shoulder = landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER]
            right_elbow = landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW]
            right_wrist = landmarks[mp_pose.PoseLandmark.RIGHT_WRIST]
            right_hip = landmarks[mp_pose.PoseLandmark.RIGHT_HIP]
            left_hip = landmarks[mp_pose.PoseLandmark.LEFT_HIP]

            # Calculate angle
            left_angle = calc_angle(left_shoulder, left_elbow, left_wrist)  # Get angle
            right_angle = calc_angle(right_shoulder, right_elbow, right_wrist)
            left_raise_angle = calc_angle(left_elbow, left_shoulder, left_hip)
            right_raise_angle = calc_angle(right_elbow, right_shoulder, right_hip)

       
            # Counter
            if left_angle > 160 and left_raise_angle < 30:
                left_flag = 'down'
            if left_angle > 160 and left_raise_angle > 80 and left_flag == 'down':
                right_count += 1
                left_flag = 'up'

            if right_angle < 160 and left_angle < 160:
                message = "straighten out left and right arm"
                cv2.rectangle(image, (5, 110), (10 + cv2.getTextSize(message, cv2.FONT_HERSHEY_SIMPLEX, 1, 2)[0][0] + 10, 100 - cv2.getTextSize(message, cv2.FONT_HERSHEY_SIMPLEX, 1, 2)[0][1] - 10), (255, 255, 255), -1), cv2.putText(image, message, (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
                cv2.putText(image, message, (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)   
            elif left_angle < 160:
                message = "straighten out right arm"
                cv2.rectangle(image, (5, 110), (10 + cv2.getTextSize(message, cv2.FONT_HERSHEY_SIMPLEX, 1, 2)[0][0] + 10, 100 - cv2.getTextSize(message, cv2.FONT_HERSHEY_SIMPLEX, 1, 2)[0][1] - 10), (255, 255, 255), -1), cv2.putText(image, message, (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
                cv2.putText(image, message, (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            elif right_angle < 160:
                message = "straighten out left arm"
                cv2.rectangle(image, (5, 110), (10 + cv2.getTextSize(message, cv2.FONT_HERSHEY_SIMPLEX, 1, 2)[0][0] + 10, 100 - cv2.getTextSize(message, cv2.FONT_HERSHEY_SIMPLEX, 1, 2)[0][1] - 10), (255, 255, 255), -1), cv2.putText(image, message, (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
                cv2.putText(image, message, (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            if right_angle > 160 and right_raise_angle < 30:
                right_flag = 'down'
            if right_angle > 160 and right_raise_angle > 80 and right_flag == 'down':
                left_count += 1
                right_flag = 'up'

            if left_count != prev_left_count or right_count != prev_right_count:
                message = "great job!"
                cv2.rectangle(image, (5, 110), (10 + cv2.getTextSize(message, cv2.FONT_HERSHEY_SIMPLEX, 1, 2)[0][0] + 10, 100 - cv2.getTextSize(message, cv2.FONT_HERSHEY_SIMPLEX, 1, 2)[0][1] - 10), (255, 255, 255), -1), cv2.putText(image, message, (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
                cv2.putText(image, message, (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                line_color = (0, 255, 0)  # Change line color to green
                green_time_start = time.time()  # Start the green timer
            elif time.time() - green_time_start > green_duration:
                line_color = (0, 0, 255)  # Change line color back to red if green duration elapsed

            prev_left_count = left_count
            prev_right_count = right_count

        except:
            pass

        # Setup Status Box
        cv2.rectangle(image, (0, 0), (1024, 73), (10, 10, 10), -1)
        cv2.putText(image, 'Left=' + str(left_count) + '           Right=' + str(right_count),
                    (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 2, cv2.LINE_AA)
        
        # Render Detections
        mp_drawing.draw_landmarks(
            image,
            results.pose_landmarks,
            mp_pose.POSE_CONNECTIONS,
            landmark_drawing_spec=mp_drawing.DrawingSpec(color=line_color, thickness=2, circle_radius=0),
            connection_drawing_spec=mp_drawing.DrawingSpec(color=line_color, thickness=2)
        )
        cv2.imshow('MAGIC', image)

        k = cv2.waitKey(30) & 0xff  # Esc for quiting the app
        if k == 27:
            break
        elif k == ord('r'):  # Reset the counter on pressing 'r' on the Keyboard
            left_count = 0
            right_count = 0

    cap.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    app.run(debug=True)