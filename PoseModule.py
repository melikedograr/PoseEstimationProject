import cv2 as cv2
import mediapipe as mp
import time


class PoseDetector:
    def __init__(self, mode=False, smoothlm=True, detection_conf=0.5, tracking_conf=0.5  ):
        self.results = None
        self.mode = mode
        self.smoothlm = smoothlm
        self.detection_conf = detection_conf
        self.tracking_conf = tracking_conf

        self.mpPose = mp.solutions.pose
        self.mpDraw = mp.solutions.drawing_utils
        self.pose = self.mpPose.Pose(self.mode, smooth_landmarks = self.smoothlm, min_detection_confidence=self.detection_conf,
               min_tracking_confidence=self.tracking_conf)


    def findPose(self, img, draw=True):
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.pose.process(imgRGB)

        if self.results.pose_landmarks:
            if draw:
                self.mpDraw.draw_landmarks(img, self.results.pose_landmarks, self.mpPose.POSE_CONNECTIONS)

        return img

    def findPosition(self, img, draw=True):
        lmList = []
        if self.results.pose_landmarks:
            for id, lm in enumerate(self.results.pose_landmarks.landmark):
                h, w, c = img.shape
                cx, cy = int(lm.x * w), int(lm.y * h)
                lmList.append([id, cx, cy])

                if draw:
                    cv2.circle(img, (cx, cy), 3, (255, 0, 255), cv2.FILLED)

        return lmList


def main():
    ptime = 0
    cap = cv2.VideoCapture('videos/video8.mp4')
    detector = PoseDetector()
    while True:
        success, img = cap.read()
        img = detector.findPose(img)
        lmList = detector.findPosition(img)
        if len(lmList) != 0:
            print(lmList[14])
            cv2.circle(img, (lmList[14][1], lmList[14][2]), 5, (0, 0, 255), cv2.FILLED)

        ctime = time.time()
        fps = 1 / (ctime - ptime)
        ptime = ctime


        cv2.putText(img, str(int(fps)), (70, 50), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 255), 3)
        cv2.imshow("Image", img)
        cv2.waitKey(1)

if __name__ == "__main__":
    main()