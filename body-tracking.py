import cv2
import numpy as np
import mediapipe as mp

class FaceAndBodyTracker:
    def __init__(self):
        self.mp_face_detection = mp.solutions.face_detection
        self.mp_pose = mp.solutions.pose
        self.face_detection = self.mp_face_detection.FaceDetection(min_detection_confidence=0.2)
        self.pose = self.mp_pose.Pose(min_detection_confidence=0.2, min_tracking_confidence=0.2)
        self.mp_draw = mp.solutions.drawing_utils

    def find_face_and_pose(self, img):
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.face_results = self.face_detection.process(img_rgb)
        self.pose_results = self.pose.process(img_rgb)
        if self.face_results.detections:
            for detection in self.face_results.detections:
                bboxC = detection.location_data.relative_bounding_box
                ih, iw, _ = img.shape
                bbox = int(bboxC.xmin * iw), int(bboxC.ymin * ih), \
                       int(bboxC.width * iw), int(bboxC.height * ih)
                cv2.rectangle(img, bbox, (255, 0, 255), 2)
        
        if self.pose_results.pose_landmarks:
            self.mp_draw.draw_landmarks(img, self.pose_results.pose_landmarks, self.mp_pose.POSE_CONNECTIONS)
        
        return img
    

def main():
    cap = cv2.VideoCapture(0)
    tracker = FaceAndBodyTracker()

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)
        frame = tracker.find_face_and_pose(frame)

        cv2.imshow("Face and Body Tracking", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
