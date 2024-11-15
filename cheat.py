import cv2
import mediapipe as mp
import numpy as np
import time

class CheatingDetector:
    def __init__(self):
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(refine_landmarks=True)
        self.detection_threshold = 5  # Maximum number of flags before detecting cheating
        self.cheating_flags = 0
        self.last_detected_time = time.time()
        self.look_away_threshold = 30  # Seconds allowed to look away before flagging
        self.frame_rate = 30  # Adjust for camera frame rate

    def calculate_gaze(self, image, landmarks):
        """
        Calculate the gaze direction based on eye landmarks.
        """
        # Left and right eye landmark indices
        left_eye_indices = [362, 385, 387, 263, 373, 380]
        right_eye_indices = [33, 160, 158, 133, 153, 144]

        h, w, _ = image.shape

        # Get the left and right eye coordinates
        left_eye = np.array([(int(landmarks[i].x * w), int(landmarks[i].y * h)) for i in left_eye_indices])
        right_eye = np.array([(int(landmarks[i].x * w), int(landmarks[i].y * h)) for i in right_eye_indices])

        # Calculate the gaze direction (using the bounding box of the eyes for simplicity)
        left_eye_center = np.mean(left_eye, axis=0).astype(int)
        right_eye_center = np.mean(right_eye, axis=0).astype(int)

        # Determine the gaze direction
        gaze_direction = "center"
        if left_eye_center[0] < w * 0.4 or right_eye_center[0] < w * 0.4:
            gaze_direction = "left"
        elif left_eye_center[0] > w * 0.6 or right_eye_center[0] > w * 0.6:
            gaze_direction = "right"

        return gaze_direction

    def detect_cheating(self):
        """
        Detect cheating based on gaze direction and head pose estimation.
        """
        cap = cv2.VideoCapture(0)
        print("Press 'q' to exit.")

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = self.face_mesh.process(rgb_frame)

            h, w, _ = frame.shape
            message = ""

            if results.multi_face_landmarks:
                # Reset timer if a face is detected
                self.last_detected_time = time.time()

                for face_landmarks in results.multi_face_landmarks:
                    # Draw the face landmarks for visualization
                    for landmark in face_landmarks.landmark:
                        x = int(landmark.x * w)
                        y = int(landmark.y * h)
                        cv2.circle(frame, (x, y), 1, (0, 255, 0), -1)

                    # Calculate gaze direction
                    gaze_direction = self.calculate_gaze(frame, face_landmarks.landmark)
                    message = f"Looking {gaze_direction.capitalize()}" if gaze_direction != "center" else ""

                    # Display message if gaze is not center
                    if gaze_direction != "center":
                        self.cheating_flags += 1
                        print(f"Cheating flag raised! Count: {self.cheating_flags}")
                    else:
                        # Reduce cheating flags if gaze returns to normal
                        self.cheating_flags = max(0, self.cheating_flags - 0.5)

            else:
                # If no face detected for a while, flag as suspicious
                if time.time() - self.last_detected_time > self.look_away_threshold:
                    self.cheating_flags += 1
                    message = "No face detected! Cheating suspected."
                    print(f"Cheating flag raised! Count: {self.cheating_flags}")
                    self.last_detected_time = time.time()

            # Display frame and message
            if message:
                cv2.putText(frame, message, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

            cv2.imshow("Cheating Detection", frame)

            # Stop detection if cheating flags exceed threshold
            if self.cheating_flags >= self.detection_threshold:
                print("Cheating detected! Alert!")
                break

            # Quit if 'q' is pressed
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()

# Run the enhanced cheating detector
detector = CheatingDetector()
detector.detect_cheating()
