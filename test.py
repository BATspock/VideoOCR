import cv2
import numpy as np
import pytesseract
import pandas as pd
import re


img = cv2.imread("test.png")

#kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
shrapened = cv2.detailEnhance(img, sigma_s=5, sigma_r=0.25)
#convert to grayscale
gray = cv2.cvtColor(shrapened, cv2.COLOR_BGR2GRAY)
#erode = cv2.erode(dilate, kernel, iterations=1)

text = pytesseract.image_to_data(gray, output_type=pytesseract.Output.DICT)
cv2.imshow("erode", gray)
if text: 
    print("text detected")
    #print(text)
    n_boxes = len(text['level'])
    
    for i in range(n_boxes):
        print(text['text'][i]) 
        if re.match(r"^.{6}[-_].[-_].{5}.*", text['text'][i]):
            (x, y, w, h) = (text['left'][i], text['top'][i], text['width'][i], text['height'][i])
            cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)

cv2.imshow("img", img)
cv2.waitKey(0)
cv2.destroyAllWindows()

# from transformers import TrOCRProcessor, VisionEncoderDecoderModel
# from PIL import Image
# import requests

# # load image from the IAM database
# url = 'https://fki.tic.heia-fr.ch/static/img/a01-122-02-00.jpg'
# image = Image.open(requests.get(url, stream=True).raw).convert("RGB")

# processor = TrOCRProcessor.from_pretrained('microsoft/trocr-base-handwritten')
# model = VisionEncoderDecoderModel.from_pretrained('microsoft/trocr-base-handwritten')
# pixel_values = processor(images=image, return_tensors="pt").pixel_values

# generated_ids = model.generate(pixel_values)
# generated_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]

