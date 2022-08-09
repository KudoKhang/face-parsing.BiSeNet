import os
import cv2
from PIL import Image
import time
import numpy as np
import torch
from ibug.face_detection import RetinaFacePredictor

class FaceDetection():
    def __init__(self, threshold=0.8, weight='mobilenet0.25'):
        self.threshold = threshold
        self.weight = weight
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.face_detector = RetinaFacePredictor(threshold=self.threshold, device=self.device,
                                                 model=(RetinaFacePredictor.get_model(self.weight)))
    def run(self, frame):
        # Detect faces
        coords = []
        faces = self.face_detector(frame, rgb=False)
        for face in faces:
            coords.append(face[:4].astype('int'))
        return coords