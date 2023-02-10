import tensorflow as tf
import pandas as pd
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import cv2

from model import Model

df = pd.read_csv("dataset.csv")

X_train, X_test, y_train, y_test = train_test_split(df.image.tolist(),df.output_map.tolist(), test_size=0.2, random_state=0)
dataset_train_X = tf.data.Dataset.from_tensor_slices(X_train)
dataset_test_X = tf.data.Dataset.from_tensor_slices(X_test)
dataset_train_y = tf.data.Dataset.from_tensor_slices(y_train)
dataset_test_y = tf.data.Dataset.from_tensor_slices(y_test)

dataset_train = tf.data.Dataset.zip((dataset_train_X, dataset_train_y))
dataset_test = tf.data.Dataset.zip((dataset_test_X, dataset_test_y))

def read_npy_file(item):
    data = np.load(item)
    return data.astype(np.float32)


def parse_image(image_filename, output_map_filename):

    image = tf.io.read_file(image_filename)
    image = tf.io.decode_png(image)
    image = tf.image.convert_image_dtype(image, tf.float32)
    output_map = tf.numpy_function(read_npy_file, [output_map_filename], [tf.float32,])
    return image, output_map


dataset_train = dataset_train.map(parse_image).batch(16, drop_remainder=True)
dataset_test = dataset_test.map(parse_image).batch(16, drop_remainder=True)

model = Model()

optimizer = tf.optimizers.SGD(learning_rate=1e-5, momentum=0.9)

def calc_confidence(map_true, map_pred):
    inter_width = tf.minimum(map_true[:,:,:,0], tf.clip_by_value(map_pred[:,:,:,0], 0, 1))
    inter_height = tf.minimum(map_true[:,:,:,1], tf.clip_by_value(map_pred[:,:,:,1], 0, 1))
    union_width = tf.maximum(map_true[:,:,:,0], tf.clip_by_value(map_pred[:,:,:,0], 0, 1))
    union_height = tf.maximum(map_true[:,:,:,1], tf.clip_by_value(map_pred[:,:,:,1], 0, 1))
    return tf.abs(inter_width*inter_height)/(tf.abs(union_width*union_height) + 1e-6)

best_eval_loss = np.inf
for epoch in range(1000):
    total_loss = 0
    for img_batch, label_batch in tqdm(dataset_train):
        label_batch = tf.reshape(label_batch, (-1,5, 5, 3))
        # orig_map = label_batch[0]

        with tf.GradientTape() as g:
            out = model(img_batch)

            confidence = calc_confidence(label_batch[:,:,:,1:],out[:,:,:,1:])
            loss1 = tf.reduce_sum(label_batch[:,:,:,0]*(confidence-out[:,:,:,0])**2)
            loss2 = 0.5*tf.reduce_sum((1-label_batch[:,:,:,0])*(confidence-out[:,:,:,0])**2)
            loss3 = 5*tf.reduce_sum(label_batch[:,:,:,0]*(label_batch[:,:,:,1]- out[:,:,:,1])**2)
            loss4 = 5*tf.reduce_sum(label_batch[:,:,:,0]*(label_batch[:,:,:,2]- out[:,:,:,2])**2)
            loss = (loss1 + loss2 + loss3 + loss4)/out.shape[0]
            total_loss += loss.numpy()
        grads = g.gradient(loss, model.trainable_weights)
        optimizer.apply_gradients(zip(grads, model.trainable_weights))

    print(f"Training: epoch = {epoch+1}, loss = {total_loss}")
    eval_loss = 0
    for img_batch, label_batch in tqdm(dataset_test):
        label_batch = tf.reshape(label_batch, (-1,5, 5, 3))
        # orig_map = label_batch[0]

        # with tf.GradientTape() as g:
        out = model(img_batch)

        confidence = calc_confidence(label_batch[:,:,:,1:],out[:,:,:,1:])
        loss1 = tf.reduce_sum(label_batch[:,:,:,0]*(confidence-out[:,:,:,0])**2)
        loss2 = 0.5*tf.reduce_sum((1-label_batch[:,:,:,0])*(confidence-out[:,:,:,0])**2)
        loss3 = 5*tf.reduce_sum(label_batch[:,:,:,0]*(label_batch[:,:,:,1]- out[:,:,:,1])**2)
        loss4 = 5*tf.reduce_sum(label_batch[:,:,:,0]*(label_batch[:,:,:,2]- out[:,:,:,2])**2)
        loss = (loss1 + loss2 + loss3 + loss4)/out.shape[0]
        eval_loss += loss.numpy()
    print(f"Evaluation: epoch = {epoch+1}, loss = {eval_loss}")
    if eval_loss <= best_eval_loss:
        print("New best eval loss")
        best_eval_loss = eval_loss
        model.save_weights("model.h5")


# del model
# model = Model()
# model.load_weights("model.h5")

img = cv2.imread(X_train[1])
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
out = model(img[None,:,:,:]/255.).numpy()[0]
for i in range(5):
    for j in range(5):
        if out[i][j][0] > 0.5*out.max():
            w = out[i][j][1]
            h = out[i][j][2]
            print(np.int0((j*224/5-(w/2*224), i*224/5-(h/2*224))),np.int0((j*224/5+(w/2*224), i*224/5+(h/2*224))))
            cv2.rectangle(img, np.int0(((j+0.5)*224/5-(w/2*224), (i+0.5)*224/5-(h/2*224))),np.int0(((j+0.5)*224/5+(w/2*224), (i+0.5)*224/5+(h/2*224))), (0, 0, 255),4)


plt.imshow(img)
plt.figure()
plt.imshow(out[:,:,1])
plt.show()



img = cv2.imread(X_test[2])
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
out = model(img[None,:,:,:]/255.).numpy()[0]

out = np.clip(out, 0, 1)
for i in range(5):
    for j in range(5):
        if out[i][j][0] > 0.5*out.max():
            w = out[i][j][1]
            h = out[i][j][2]
            print(np.int0((j*224/5-(w/2*224), i*224/5-(h/2*224))),np.int0((j*224/5+(w/2*224), i*224/5+(h/2*224))))
            cv2.rectangle(img, np.int0(((j+0.5)*224/5-(w/2*224), (i+0.5)*224/5-(h/2*224))),np.int0(((j+0.5)*224/5+(w/2*224), (i+0.5)*224/5+(h/2*224))), (0, 0, 255),4)


plt.imshow(img)
plt.figure()
plt.imshow(out[:,:,1])
plt.show()