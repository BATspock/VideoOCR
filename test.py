import cv2
import numpy as np
import pytesseract
import pandas as pd
import re


# read frame from videos
# identify if text is present in the frame
# if text is present, create a bounding box around the frame

# video = "Videos/071004-F-2184C-001.mov"

# cap = cv2.VideoCapture(video)

img = cv2.imread("test.png")

gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#sharpen the image


thresh = cv2.threshold(gray,0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]

kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
shrapened = cv2.filter2D(thresh, -1, kernel)

#erode = cv2.erode(dilate, kernel, iterations=1)

text = pytesseract.image_to_data(shrapened, output_type=pytesseract.Output.DICT)
cv2.imshow("erode", shrapened)
if text: 
    print("text detected")
    print(text)
    n_boxes = len(text['level'])
    for i in range(n_boxes):
        if int(text['conf'][i]) > 60 and re.match(r"\d+[_-][a-zA-Z][_-]\d+[a-zA-Z][-_]\d+", text['text'][i]):
            (x, y, w, h) = (text['left'][i], text['top'][i], text['width'][i], text['height'][i])
            cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)

cv2.imshow("img", img)
cv2.waitKey(0)
cv2.destroyAllWindows()
