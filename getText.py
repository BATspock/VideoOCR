import cv2
import numpy as np
import pytesseract
import pandas as pd
import re


# read frame from videos
# identify if text is present in the frame
# if text is present, create a bounding box around the text


# video = "Videos/071004-F-2184C-001.mov"


def find_text(video):
    flag = False
    # get name of video
    video_name = video.split("/")[-1].split(".")[0]
    cap = cv2.VideoCapture(video)

    df = pd.DataFrame(columns=["timestamp", "image name", "text", "Confidence"])
    prev_frame = None
    while cap.isOpened():
        # print("Frame: {}".format(cap.get(cv2.CAP_PROP_POS_FRAMES)))
        ret, frame = cap.read()

        if ret:
            # frame = cv2.detailEnhance(frame, sigma_s=5, sigma_r=0.25)

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            text = pytesseract.image_to_data(gray, output_type=pytesseract.Output.DICT)

            if text:
                timestamp = cap.get(cv2.CAP_PROP_POS_MSEC)
                n_boxes = len(text["level"])
                for i in range(n_boxes):
                    # if int(text['conf'][i]) > 60 and re.match(r"\d+[_-][a-zA-Z][_-]\d+[a-zA-Z][-_]\d+", text['text'][i]):
                    if re.match(r"^.{6}[-_].[-_].{5}.*", text["text"][i]):
                        flag = True
                        (x, y, w, h) = (
                            text["left"][i],
                            text["top"][i],
                            text["width"][i],
                            text["height"][i],
                        )
                        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                        if prev_frame is None:
                            prev_frame = frame
                            cv2.imwrite(
                                "frames/"
                                + video_name
                                + "_frame_{}.png".format(timestamp),
                                frame,
                            )
                            df = df.append(
                                {
                                    "timestamp": timestamp,
                                    "image name": video_name
                                    + "_frame_{}.png".format(timestamp),
                                    "text": text["text"][i],
                                    "Confidence": text["conf"][i],
                                },
                                ignore_index=True,
                            )

                        elif not np.equal(prev_frame, frame).all():
                            prev_frame = frame
                            cv2.imwrite(
                                "frames/"
                                + video_name
                                + "_frame_{}.png".format(timestamp),
                                frame,
                            )
                            df = df.append(
                                {
                                    "timestamp": timestamp,
                                    "image name": video_name
                                    + "_frame_{}.png".format(timestamp),
                                    "text": text["text"][i],
                                    "Confidence": text["conf"][i],
                                },
                                ignore_index=True,
                            )
        else:
            break

    if flag:
        print("Text detected in {}".format(video_name))
        df.to_csv("CSV/" + video_name + "_output.csv", index=False)
    else:
        print("No VIRIN detected in {}".format(video_name))
        
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    find_text("Videos/071004-F-2184C-001.mov")
