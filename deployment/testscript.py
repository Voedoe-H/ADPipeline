import numpy as np
import base64
import json
import requests
import os
import cv2

data_path_good = os.path.abspath("../data/raw/screw/test/good/000.png")
data_path_bad = os.path.abspath("../data/raw/screw/test/manipulated_front/000.png")

img = cv2.imread(data_path_good, cv2.IMREAD_GRAYSCALE)

if img.shape != (1024, 1024):
    img = cv2.resize(img, (1024, 1024))

img_bytes = img.tobytes()
img_b64 = base64.b64encode(img_bytes).decode('utf-8')

payload = {
    "image": img_b64
}

#print(img_b64) 

response = requests.post("http://localhost:8080/infer", json=payload)
print(response.text)
