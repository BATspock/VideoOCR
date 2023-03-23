import cv2
import numpy as np

frame1 = cv2.imread('frames/frame_0.0.png',0)
frame2 = cv2.imread('frames/frame_33.366700033366705.png',0)
if np.equal(frame1, frame2).all():
    print("equal")