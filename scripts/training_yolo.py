#%%
import os,sys
sys.path.insert(0,os.path.abspath(os.path.join(os.path.dirname(__file__),'..')))
from utils.basic_functions import get_secret,get_image
from ultralytics import YOLO
# %%
import requests,cv2
import numpy as np
import matplotlib.pyplot as plt
model = YOLO('yolov8l.pt')
url = 'https://cumberland.isis.vanderbilt.edu/stefan/rvlcdip/test_advertisement_127238fe-8352-49ed-8c5f-11488a154dd2.jpg'
response = requests.get(url,stream=True).raw
image = np.asarray(bytearray(response.read()),dtype=np.uint8)
image = cv2.imdecode(image,cv2.IMREAD_COLOR)
results = model(image)
print(len(results))

for result in results:
    for box in result.boxes.xyxy:
        x_min,y_min,x_max,y_max = map(int,box[:4])
        cv2.rectangle(image,(x_min,y_min),(x_max,y_max),(0,255,0),2)

plt.imshow(cv2.cvtColor(image,cv2.COLOR_BGR2RGB))
plt.axis('off')
plt.show()
# %%
