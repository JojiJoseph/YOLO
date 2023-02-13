import argparse
import numpy as np
import cv2

from model import Model

categories = ["aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car", "cat", "chair", "cow", "dog", "horse", "motorbike", "person", "sheep", "sofa", "diningtable", "pottedplant", "train", "tvmonitor"]


model = Model()
model(np.zeros((1,224,224,3)))
model.load_weights("model.h5")

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break
    print(frame.shape)
    processed_frame = cv2.resize(frame, (224,224))
    processed_frame = cv2.cvtColor(processed_frame, cv2.COLOR_BGR2RGB)
    out = model(processed_frame[None,...].astype(float)/127.5-1, training=False)[0].numpy()
    height, width, _ = frame.shape
    print(out[:,:,0].max())
    cv2.imshow("in", frame)
    for i in range(5):
        for j in range(5):
            if out[i][j][0] >= 0.5:# 0.8 * out[:,:,0].max() and out[:,:,0].max()>0.3:
                w = out[i][j][1]
                h = out[i][j][2]
                x_offset = out[i][j][3]
                y_offset = out[i][j][4]
                category = categories[np.argmax(out[i][j][5:])]
                print(category)
                print(np.int0((j*224/5-(w/2*224), i*224/5-(h/2*224))),np.int0((j*224/5+(w/2*224), i*224/5+(h/2*224))))
                cv2.rectangle(frame, np.int0(((j+x_offset)*width/5-(w/2*width), (i+y_offset)*height/5-(h/2*height))),np.int0(((j+x_offset)*width/5+(w/2*width), (i+y_offset)*height/5+(h/2*height))), (0, 0, 255),4)
                cv2.putText(frame, category, np.int0(((j+x_offset)*width/5-(w/2*width), (i+0.5)*height/5-(h/2*height))),cv2.FONT_HERSHEY_COMPLEX,1,(255,0,0))
    cv2.imshow("out", frame)
    key = cv2.waitKey(10)
    if key in [27, ord('q')]:
        break