import cv2 
import mediapipe as mp
import time

class FaceMeshDetector():
    def __init__(self, staticMode = False, maxFaces = 2, minDetectionCon = False, minTrackCon = 0.5):
        self.staticMode = staticMode
        self.maxFaces = maxFaces
        self.minDetectionCon = minDetectionCon
        self.minTrackCon = minTrackCon

        self.mpDraw = mp.solutions.drawing_utils         
        self.mpFaceMesh = mp.solutions.face_mesh
        self.faceMesh = self.mpFaceMesh.FaceMesh(self.staticMode, self.maxFaces,
                                                self.minDetectionCon, self.minTrackCon )
        self.drawSpec = self.mpDraw.DrawingSpec(thickness=1, circle_radius = 1)

    def findFaceMesh(self, img, draw = True):
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.faceMesh.process(imgRGB)
        faces = []
        if self.results.multi_face_landmarks:
            for faceLms in self.results.multi_face_landmarks:
                self.mpDraw.draw_landmarks(img, faceLms, self.mpFaceMesh.FACEMESH_CONTOURS, self.drawSpec, self.drawSpec)
                face = []
                for id, lm in enumerate(faceLms.landmark):
                    h, w, c = img.shape
                    x, y = int(lm.x*w), int(lm.y*h)
                    # cv2.putText(img, str(id), (x,y), cv2.FONT_HERSHEY_PLAIN, 1,  (255,255,255), 1)
                    # print(id, x, y)
                    face.append([x,y])
                faces.append(face)
        return img , faces

   
def main():
    cap = cv2.VideoCapture("twofaces.mp4")
    pTime = 0
    detector = FaceMeshDetector()
    while True:
        success, img = cap.read()
        img , faces = detector.findFaceMesh(img)
        if len(faces)!=0:
            print(len(faces))
        try:
            cTime = time.time()
            fps = 1 / (cTime - pTime)
            pTime = cTime
            cv2.putText(img, f'FPS: {str(int(fps))}', (10,40), cv2.FONT_HERSHEY_PLAIN, 2,  (0,255,0), 2)
        except ZeroDivisionError:
            fps = 0
        cv2.imshow("Image",img)
        cv2.waitKey(1)

if __name__ == "__main__":
    main()