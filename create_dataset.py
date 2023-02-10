import os
from PIL import Image
import xml.etree.ElementTree as ET
import cv2
import numpy as np
from tqdm import tqdm
import pandas as pd

image_path = []
output_map_path = []

files  = os.listdir("dataset/Annotations")

import matplotlib.pyplot as plt




for filename in tqdm(files):
    if filename.endswith(".xml"):
        objmap = [[[0,0,0] for j in range(5)] for _ in range(5)]
        part_name = filename[:-4]
        et = ET.parse(os.path.join("dataset/Annotations", filename))
        objects = et.getroot().findall('object')
        size = et.getroot().find('size')
        width = int(size.find('width').text)
        height = int(size.find('height').text)
        img = Image.open(os.path.join("dataset/JPEGImages", part_name + ".jpg"))
        img = np.asarray(img)
        img_out = img.copy()

        for object in objects:
            for child in object:
                # print(object.find('name').text)
                if child.tag == "bndbox":
                    # print(child.find('xmin').text)
                    xmin = int(child.find('xmin').text)
                    xmax = int(child.find('xmax').text)
                    ymin = int(child.find('ymin').text)
                    ymax = int(child.find('ymax').text)
                    xmid = (xmin+xmax)/2
                    ymid = (ymin+ymax)/2
                    objmap[int(ymid*5/height)][int(xmid*5/width)][0] = 1.0
                    objmap[int(ymid*5/height)][int(xmid*5/width)][1] = (xmax-xmin)/width
                    objmap[int(ymid*5/height)][int(xmid*5/width)][2] = (ymax-ymin)/height
                    # cv2.rectangle(img, (xmin, ymin), (xmax, ymax), (0,0,255),2)
        cv2.imwrite(os.path.join("dataset/Preprocessed", part_name + ".png"),cv2.resize(img_out, (224, 224)))
        np.save(os.path.join("dataset/OutputMap", part_name + ".npy"),np.array(objmap))
        for i in range(5):
            for j in range(5):
                if objmap[i][j][0] > 0:
                    x1, y1 = (j+0.5)*width/5 - objmap[i][j][1]*width/2, (i+0.5)*height/5 - objmap[i][j][2]*height/2
                    x2, y2 = (j+0.5)*width/5 + objmap[i][j][1]*width/2, (i+0.5)*height/5 + objmap[i][j][2]*height/2
                    print(x1, y1, x2, y2)
                    cv2.rectangle(img_out, np.int0((x1, y1)), np.int0((x2, y2)),(255, 0, 0))
        plt.imshow(img_out)
        plt.figure()
        plt.imshow(np.array(objmap)[:,:,0])
        plt.show()
        image_path.append(os.path.join("dataset/Preprocessed", part_name + ".png"))
        output_map_path.append(os.path.join("dataset/OutputMap", part_name + ".npy"))

df = pd.DataFrame({"image": image_path, "output_map": output_map_path })
df.to_csv("dataset.csv", index=False)