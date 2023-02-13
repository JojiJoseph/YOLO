import tensorflow as tf
import pandas as pd
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import cv2
from tensorflow.python.ops.numpy_ops import np_config
np_config.enable_numpy_behavior()

from model import Model

categories = ["aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car", "cat", "chair", "cow", "dog", "horse", "motorbike", "person", "sheep", "sofa", "diningtable", "pottedplant", "train", "tvmonitor"]


df = pd.read_csv("dataset.csv")

X_train, X_test, y_train, y_test = train_test_split(df.image.tolist(),df.output_map.tolist(), test_size=0.2, random_state=0)
dataset_train_X = tf.data.Dataset.from_tensor_slices(X_train[:])
dataset_test_X = tf.data.Dataset.from_tensor_slices(X_test[:])
dataset_train_y = tf.data.Dataset.from_tensor_slices(y_train[:])
dataset_test_y = tf.data.Dataset.from_tensor_slices(y_test[:])

dataset_train = tf.data.Dataset.zip((dataset_train_X, dataset_train_y))
dataset_test = tf.data.Dataset.zip((dataset_test_X, dataset_test_y))

def read_npy_file(item):
    data = np.load(item)
    return data.astype(np.float32)


def parse_image(image_filename, output_map_filename):

    image = tf.io.read_file(image_filename)
    image = tf.io.decode_png(image)
    image = tf.image.convert_image_dtype(image, tf.float32) * 2 - 1
    output_map = tf.numpy_function(read_npy_file, [output_map_filename], [tf.float32,])
    return image, output_map


dataset_train = dataset_train.map(parse_image).batch(16, drop_remainder=True)
dataset_test = dataset_test.map(parse_image).batch(16, drop_remainder=True)

model = Model()

optimizer = tf.optimizers.SGD(learning_rate=2e-5, momentum=0.1)

def calc_confidence(map_true, map_pred, epoch):
    if epoch < 50:
        inter_width = tf.minimum(map_true[:,:,:,1], tf.clip_by_value(map_pred[:,:,:,1], 0, 1))
        inter_height = tf.minimum(map_true[:,:,:,2], tf.clip_by_value(map_pred[:,:,:,2], 0, 1))
        union_width = tf.maximum(map_true[:,:,:,1], tf.clip_by_value(map_pred[:,:,:,1], 0, 1))
        union_height = tf.maximum(map_true[:,:,:,2], tf.clip_by_value(map_pred[:,:,:,2], 0, 1))
        iou = inter_width* inter_height / (union_width * union_height + 1e-6)
        iou = tf.clip_by_value(iou, 0, 1)
        return iou
    scale_matrix_x = np.matrix([[0, 1,2,3,4]]*5)
    scale_matrix_y = scale_matrix_x.T
    x1_true = scale_matrix_x/5 + map_true[:,:,:,3]/5 - map_true[:,:,:,1]/2
    x1_pred = scale_matrix_x/5 + map_pred[:,:,:,3]/5 - map_pred[:,:,:,1]/2
    x2_true = scale_matrix_x/5 + map_true[:,:,:,3]/5 + map_true[:,:,:,1]/2
    x2_pred = scale_matrix_x/5 + map_pred[:,:,:,3]/5 + map_pred[:,:,:,1]/2
    y1_true = scale_matrix_y/5 + map_true[:,:,:,4]/5 - map_true[:,:,:,2]/2
    y1_pred = scale_matrix_y/5 + map_pred[:,:,:,4]/5 - map_pred[:,:,:,2]/2
    y2_true = scale_matrix_y/5 + map_true[:,:,:,4]/5 + map_true[:,:,:,2]/2
    y2_pred = scale_matrix_y/5 + map_pred[:,:,:,4]/5 + map_pred[:,:,:,2]/2

    inter_width = (x1_true < x1_pred).astype(float) * (x2_true - x1_pred) + (x1_true >= x1_pred).astype(float) * (x2_pred - x1_true)
    inter_height = (y1_true < y1_pred).astype(float) * (y2_true - y1_pred) + (y1_true >= y1_pred).astype(float) * (y2_pred - y1_true)

    inter_width = tf.clip_by_value(inter_width, 0, 1)
    inter_height = tf.clip_by_value(inter_height, 0, 1)
    inter_area = inter_width * inter_height
    union_area = tf.abs(map_pred[:,:,:,1] * map_pred[:,:,:,2]) + tf.abs(map_true[:,:,:,1] * map_true[:,:,:,2]) - inter_area
    union_area = tf.clip_by_value(union_area, 0, 1)
    iou = tf.abs(inter_area)/(tf.abs(union_area) +    1e-6)

    iou = (iou <= 1).astype(float) * iou
    return tf.clip_by_value(iou, 0, 1)

best_eval_loss = np.inf
for epoch in range(500):
    total_loss = 0
    for img_batch, label_batch in tqdm(dataset_train):
        label_batch = tf.reshape(label_batch, (-1,5, 5, 25))

        with tf.GradientTape() as g:
            out = model(img_batch)

            confidence = calc_confidence(label_batch[:,:,:,:],out[:,:,:,:], epoch)
            loss1 = tf.reduce_sum(label_batch[:,:,:,0]*(confidence-out[:,:,:,0])**2)
            loss2 = 0.5*tf.reduce_sum((1-label_batch[:,:,:,0])*(confidence-out[:,:,:,0])**2)
            loss3 = 5*tf.reduce_sum(label_batch[:,:,:,0]*(label_batch[:,:,:,1]- out[:,:,:,1])**2)
            loss4 = 5*tf.reduce_sum(label_batch[:,:,:,0]*(label_batch[:,:,:,2]- out[:,:,:,2])**2)
            loss5 = 5*tf.reduce_sum(label_batch[:,:,:,0]*(label_batch[:,:,:,3]- out[:,:,:,3])**2)
            loss6 = 5*tf.reduce_sum(label_batch[:,:,:,0]*(label_batch[:,:,:,4]- out[:,:,:,4])**2)

            loss7 = tf.reduce_sum(label_batch[:,:,:,0:1]*(label_batch[:,:,:,5:]- out[:,:,:,5:])**2)
            loss = (loss1 + loss2 + loss3 + loss4 + loss5  + loss6 + loss7)/out.shape[0]
            total_loss += loss.numpy()
        grads = g.gradient(loss, model.trainable_weights)
        optimizer.apply_gradients(zip(grads, model.trainable_weights))

    print(f"Training: epoch = {epoch+1}, loss = {total_loss}")
    eval_loss = 0
    for img_batch, label_batch in tqdm(dataset_test):

        label_batch = tf.reshape(label_batch, (-1,5, 5, 25))

        out = model(img_batch)

        confidence = calc_confidence(label_batch[:,:,:,:],out[:,:,:,:], epoch)
        loss1 = tf.reduce_sum(label_batch[:,:,:,0]*(confidence-out[:,:,:,0])**2)
        loss2 = 0.5*tf.reduce_sum((1-label_batch[:,:,:,0])*(confidence-out[:,:,:,0])**2)
        loss3 = 5*tf.reduce_sum(label_batch[:,:,:,0]*(label_batch[:,:,:,1]- out[:,:,:,1])**2)
        loss4 = 5*tf.reduce_sum(label_batch[:,:,:,0]*(label_batch[:,:,:,2]- out[:,:,:,2])**2)
        loss5 = 5*tf.reduce_sum(label_batch[:,:,:,0]*(label_batch[:,:,:,3]- out[:,:,:,3])**2)
        loss6 = 5*tf.reduce_sum(label_batch[:,:,:,0]*(label_batch[:,:,:,4]- out[:,:,:,4])**2)
        loss7 = tf.reduce_sum(label_batch[:,:,:,0:1]*(label_batch[:,:,:,5:]- out[:,:,:,5:])**2)
        loss = (loss1 + loss2 + loss3 + loss4 + loss5  + loss6 + loss7)/out.shape[0]
        eval_loss += loss.numpy()
    print(f"Evaluation: epoch = {epoch+1}, loss = {eval_loss}")
    if eval_loss <= best_eval_loss:
        print("New best eval loss")
        best_eval_loss = eval_loss
        model.save_weights("model.h5")

    if epoch == 1:
        optimizer = tf.optimizers.Adam(learning_rate=2e-4)

    if epoch == 20:
        optimizer = tf.optimizers.Adam(learning_rate=2e-5)
    if epoch == 50:
        optimizer = tf.optimizers.Adam(learning_rate=2e-6)


# To test
img = cv2.imread(X_train[1])
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
out = model(img[None,:,:,:]/127.5-1).numpy()[0]
for i in range(5):
    for j in range(5):
        if out[i][j][0] > 0.6:#*out.max():
            w = out[i][j][1]
            h = out[i][j][2]
            x_offset = out[i][j][3]
            y_offset = out[i][j][4]
            category = categories[np.argmax(out[i][j][5:])]
            print(category)
            cv2.rectangle(img, np.int0(((j+x_offset)*224/5-(w/2*224), (i+y_offset)*224/5-(h/2*224))),np.int0(((j+x_offset)*224/5+(w/2*224), (i+y_offset)*224/5+(h/2*224))), (0, 0, 255),4)
            cv2.putText(img, category, np.int0(((j+x_offset)*224/5-(w/2*224), (i+y_offset)*224/5-(h/2*224))),cv2.FONT_HERSHEY_COMPLEX,1,(255,0,0))

plt.imshow(img)
plt.figure()
plt.imshow(out[:,:,0])
plt.show()



img = cv2.imread(X_test[2])
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
out = model(img[None,:,:,:]/127.5-1., training=False).numpy()[0]

out = np.clip(out, 0, 1)
for i in range(5):
    for j in range(5):
        if out[i][j][0] > 0.6:#*out.max():
            w = out[i][j][1]
            h = out[i][j][2]
            x_offset = out[i][j][3]
            y_offset = out[i][j][4]
            category = categories[np.argmax(out[i][j][5:])]
            print(category)
            cv2.rectangle(img, np.int0(((j+x_offset)*224/5-(w/2*224), (i+y_offset)*224/5-(h/2*224))),np.int0(((j+x_offset)*224/5+(w/2*224), (i+y_offset)*224/5+(h/2*224))), (0, 0, 255),4)
            cv2.putText(img, category, np.int0(((j+x_offset)*224/5-(w/2*224), (i+0.5)*224/5-(h/2*224))),cv2.FONT_HERSHEY_COMPLEX,1,(255,0,0))

plt.imshow(img)
plt.figure()
plt.imshow(out[:,:,0])
plt.show()
