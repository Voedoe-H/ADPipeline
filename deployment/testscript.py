import numpy as np
import base64
import json
import requests

img = np.random.randint(0, 256, (1024, 1024), dtype=np.uint8)

img_bytes = img.tobytes()
img_b64 = base64.b64encode(img_bytes).decode('utf-8')

payload = {
    "image": img_b64
}

print(img_b64) 

response = requests.post("http://localhost:8080/infer", json=payload)
print(response.text)
