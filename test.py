from modules import FaceDetection
import cv2

facedetection = FaceDetection()

image = cv2.imread('src/image_test/1.jpg')
face = facedetection.run(image) # x1,y1,x2,y2
if len(face) == 0:
    print('No face detection')
if len(face) == 1:
    coord = face[0]
print(coord)