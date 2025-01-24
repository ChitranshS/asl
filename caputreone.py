import cv2
from pathlib import Path
import time
import os

BASE_DIR = Path(__file__).parent
IMAGE_PATH = "DataCollection"
data_path = os.path.join(BASE_DIR,IMAGE_PATH)
if not os.path.exists(data_path):
    os.makedirs(data_path)


cap = cv2.VideoCapture(0)
for num in range(3):
    ret, frame = cap.read()
    if not ret:
        break
    # image_name = os.path.join(img_path, label + "." + "{}.jpg".format(num))
    time.sleep(2)
    image_name = os.path.join(data_path, str(num) + ".jpg")
    cv2.imwrite(image_name,frame)
    cv2.imshow('frame', frame)
    

    if cv2.waitKey(1) & 0xFF== ord('q'):
        break