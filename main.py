import cv2
import numpy as np
import pytesseract
import pandas as pd
import re


# read frame from videos
# identify if text is present in the frame
# if text is present, create a bounding box around the text

video = "Videos/071004-F-2184C-001.mov"
df = pd.DataFrame(columns=['timestamp', 'image name', 'text', 'Confidence'])
cap = cv2.VideoCapture(video)
prev_frame = None
while cap.isOpened():
    print("Frame: {}".format(cap.get(cv2.CAP_PROP_POS_FRAMES)))
    ret, frame = cap.read()
    if ret:
        frame = cv2.detailEnhance(frame, sigma_s=5, sigma_r=0.25)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        #sharpen the image


        #erode = cv2.erode(dilate, kernel, iterations=1)

        text = pytesseract.image_to_data(gray, output_type=pytesseract.Output.DICT)
        if text: 
            timestamp = cap.get(cv2.CAP_PROP_POS_MSEC)
            n_boxes = len(text['level'])
            for i in range(n_boxes):
                if int(text['conf'][i]) > 60 and re.match(r"\d+[_-][a-zA-Z][_-]\d+[a-zA-Z][-_]\d+", text['text'][i]):
                    (x, y, w, h) = (text['left'][i], text['top'][i], text['width'][i], text['height'][i])
                    cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                    if prev_frame is None:
                        prev_frame = frame
                        cv2.imwrite("frames/frame_{}.png".format(timestamp), frame)
                        df = df.append({'timestamp': timestamp, 'image name': "frame_{}.png".format(timestamp), 'text': text['text'][i], 'Confidence': text['conf'][i]}, ignore_index=True)

                    elif not np.equal(prev_frame, frame).all():
                        prev_frame = frame
                        cv2.imwrite("frames/frame_{}.png".format(timestamp), frame)
                        df = df.append({'timestamp': timestamp, 'image name': "frame_{}.png".format(timestamp), 'text': text['text'][i], 'Confidence': text['conf'][i]}, ignore_index=True)
    else:
        break
df.to_csv("output.csv", index=False)
cap.release()
cv2.destroyAllWindows()
