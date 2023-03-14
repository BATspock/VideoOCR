import cv2
import numpy as np
import pytesseract
import pandas as pd

def detect_text(img):
    text = pytesseract.image_to_string(img)
    return text

def convert_to_gray(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return gray

def remove_noise(img):
    return cv2.medianBlur(img, 5)

def thresholding(img):
    return cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
# capture video from a link and save timestamps in a file when text is detected
url = 'Videos/071004-F-2184C-001.mov'
cap = cv2.VideoCapture(url)



while True:

    ret, frame = cap.read()
    if not ret:
        break

    gray = convert_to_gray(frame)

    # custom_config = r'--oem 3 --psm 6'

    text = pytesseract.image_to_data(gray, output_type=pytesseract.Output.DICT)
    if text:
        timestamp = cap.get(cv2.CAP_PROP_POS_MSEC)
        print(f'Text found at timestamp {timestamp:.2f} ms')
        # print(text)
        # print(text['conf'])
        text['conf'] = [int(i) for i in text['conf']]
        df = pd.DataFrame({'text': text['text'], 'conf': text['conf']})
        df = df[df['conf'] > 50]
        # add timestamp to the dataframe
        df['timestamp'] = timestamp
    
        # save the dataframe to a csv file
        df.to_csv('text.csv', mode='a', header=False, index=False)
