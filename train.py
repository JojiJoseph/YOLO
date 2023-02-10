import tensorflow as tf
import pandas as pd
import numpy as np
from tqdm import tqdm
from sklearn.model_selection import train_test_split

from model import Model

df = pd.read_csv("dataset.csv")

X_train, X_test, y_train, y_test = train_test_split(df.image.tolist(),df.output_map.tolist(), test_size=0.2, random_state=0)
dataset_train_X = tf.data.Dataset.from_tensor_slices(X_train)
dataset_test_X = tf.data.Dataset.from_tensor_slices(X_test)
dataset_train_y = tf.data.Dataset.from_tensor_slices(y_train)
dataset_test_y = tf.data.Dataset.from_tensor_slices(y_test)

dataset_train = tf.data.Dataset.zip((dataset_train_X, dataset_train_y))
dataset_test = tf.data.Dataset.zip((dataset_test_X, dataset_test_y))
# print(df.image.tolist()[:100])
# exit()
# dataset = tf.data.Dataset.from_tensor_slices(list(zip(df.image.tolist(), df.output_map.tolist())))
def read_npy_file(item):
    data = np.load(item)
    return data.astype(np.float32)

# file_list = ['/foo/bar.npy', '/foo/baz.npy']

# dataset = tf.data.Dataset.from_tensor_slices(file_list)

# dataset = dataset.map(
#         lambda item: tuple(tf.py_func(read_npy_file, [item], [tf.float32,])))
# tf.enable_eager_execution()

def parse_image(image_filename, output_map_filename):
    # image_filename, output_map_filename = names
#   parts = tf.strings.split(filename, os.sep)
#   label = parts[-2]
    # print(image_filename)

    image = tf.io.read_file(image_filename)
    image = tf.io.decode_png(image)
    image = tf.image.convert_image_dtype(image, tf.float32)#/127.5-1.
    # print(tf.reduce_min(image).numpy(), tf.reduce_max(image))
    # print(image.shape)
    # output_map = tf.io.read_file(output_map_filename)
    # output_map.
    # output_map = tf.io.decode_raw(output_map.numpy(), tf.float32)
    # print(str(output_map_filename.numpy().decode("utf-8") ))
    # output_map_filename = output_map_filename.numpy().decode("utf-8")
    # print(output_map_filename.numpy(), "$")
    output_map = tf.numpy_function(read_npy_file, [output_map_filename], [tf.float32,])
    # output_map = read_npy_file(output_map_filename)

    # print(output_map.shape)
    return image, output_map

# (parse_image(next(iter(dataset))))

dataset_train = dataset_train.map(parse_image).batch(16, drop_remainder=True)
dataset_test = dataset_test.map(parse_image).batch(16, drop_remainder=True)

model = Model()

# model.compile(optimizer='adam', loss=tf.keras.losses.BinaryCrossentropy(from_logits=True))

# model.fit(dataset,batch_size=64)

# optimizer = tf.optimizers.Adam(learning_rate=3e-5)
optimizer = tf.optimizers.SGD(learning_rate=1e-5, momentum=0.9)

def calc_confidence(map_true, map_pred):
    inter_width = tf.minimum(map_true[:,:,:,0], tf.clip_by_value(map_pred[:,:,:,0], 0, 1))
    inter_height = tf.minimum(map_true[:,:,:,1], tf.clip_by_value(map_pred[:,:,:,1], 0, 1))
    union_width = tf.maximum(map_true[:,:,:,0], tf.clip_by_value(map_pred[:,:,:,0], 0, 1))
    union_height = tf.maximum(map_true[:,:,:,1], tf.clip_by_value(map_pred[:,:,:,1], 0, 1))
    return tf.abs(inter_width*inter_height)/(tf.abs(union_width*union_height) + 1e-6)

step = 0
for epoch in range(10):
    total_loss = 0
    for img_batch, label_batch in tqdm(dataset_train):
        # print(img_batch.numpy().min(), img_batch.numpy().max())
        # print(label_batch.shape)
        label_batch = tf.reshape(label_batch, (-1,5, 5, 3))
        orig_map = label_batch[0]

        with tf.GradientTape() as g:
            out = model(img_batch)

            confidence = calc_confidence(label_batch[:,:,:,1:],out[:,:,:,1:])
            # print(confidence.shape, tf.reduce_min(confidence), tf.reduce_max(confidence))
            # print(label_batch[:,:,:,0].shape,tf.keras.losses.mean_squared_error(label_batch[:,:,:,0], out[:,:,:,0]).shape)
            # label_objects = tf.reshape(label_batch[:,:,:,0], (out.shape[0],-1))
            # out_objects = tf.reshape(out[:,:,:,0], (out.shape[0],-1))
            # loss1 = tf.reduce_sum(label_batch[:,:,:,0]*(label_batch[:,:,:,0]-out[:,:,:,0])**2)
            # loss2 = 0.5*tf.reduce_sum((1-label_batch[:,:,:,0])*(label_batch[:,:,:,0]-out[:,:,:,0])**2)
            loss1 = tf.reduce_sum(label_batch[:,:,:,0]*(confidence-out[:,:,:,0])**2)
            loss2 = 0.5*tf.reduce_sum((1-label_batch[:,:,:,0])*(confidence-out[:,:,:,0])**2)
            loss3 = 5*tf.reduce_sum(label_batch[:,:,:,0]*(label_batch[:,:,:,1]- out[:,:,:,1])**2)
            loss4 = 5*tf.reduce_sum(label_batch[:,:,:,0]*(label_batch[:,:,:,2]- out[:,:,:,2])**2)
            # print(out.shape)
            # print(label_batch[:,:,:,0])
            # print(label_batch[:,:,:,2].shape,out[:,:,:,2].shape, out.shape[0])
            loss = (loss1 + loss2 + loss3 + loss4)/out.shape[0]
            # print(loss)
            total_loss += loss.numpy()
        grads = g.gradient(loss, model.trainable_weights)
        optimizer.apply_gradients(zip(grads, model.trainable_weights))
        step += 1
        # if step > 10:
        #     break
    print(epoch, total_loss)

# optimizer = tf.optimizers.SGD(learning_rate=1e-5, momentum=0.9)
# for epoch in range(0):
#     total_loss = 0
#     for img_batch, label_batch in tqdm(dataset):
#         # print(label_batch.shape)
#         label_batch = tf.reshape(label_batch, (-1,5, 5, 3))

#         with tf.GradientTape() as g:
#             out = model(img_batch)

#             confidence = calc_confidence(label_batch[:,:,:,1:],out[:,:,:,1:])
#             # print(confidence.shape, tf.reduce_min(confidence), tf.reduce_max(confidence))
#             # print(label_batch[:,:,:,0].shape,tf.keras.losses.mean_squared_error(label_batch[:,:,:,0], out[:,:,:,0]).shape)
#             # label_objects = tf.reshape(label_batch[:,:,:,0], (out.shape[0],-1))
#             # out_objects = tf.reshape(out[:,:,:,0], (out.shape[0],-1))
#             # loss1 = tf.reduce_sum(label_batch[:,:,:,0]*(label_batch[:,:,:,0]-out[:,:,:,0])**2)
#             # loss2 = 0.5*tf.reduce_sum((1-label_batch[:,:,:,0])*(label_batch[:,:,:,0]-out[:,:,:,0])**2)
#             loss1 = tf.reduce_sum(label_batch[:,:,:,0]*(confidence-out[:,:,:,0])**2)
#             loss2 = 0.5*tf.reduce_sum((1-label_batch[:,:,:,0])*(confidence-out[:,:,:,0])**2)
#             loss3 = 5*tf.reduce_sum(label_batch[:,:,:,0]*(label_batch[:,:,:,1]- out[:,:,:,1]**2)**2)
#             loss4 = 5*tf.reduce_sum(label_batch[:,:,:,0]*(label_batch[:,:,:,2]- out[:,:,:,2]**2)**2)
#             loss = (loss1 + loss2 + loss3 + loss4)/out.shape[0]
#             print(loss)
#             total_loss += loss.numpy()
#         grads = g.gradient(loss, model.trainable_weights)
#         optimizer.apply_gradients(zip(grads, model.trainable_weights))
#         step += 1
#         # if step > 10:
#         #     break
#     print(epoch+1, total_loss)

import cv2
# orig_map = orig_map.numpy()
img = cv2.imread(X_train[3])#"dataset/Preprocessed/000019.png")#df.image.tolist()[0])
# img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
out = model(img[None,:,:,::-1]/255.).numpy()[0]
# print(np.mean(img/255.), img.min())
# exit()
print(out[:,:,0])
# out = np.clip(out, 0, 1)
out = np.clip(out, 0, 1)
print(out[:,:,0])
# for i in range(5):
#     for j in range(5):
#         if orig_map[i][j][0] > 0:
#             w = out[i][j][1]
#             h = out[i][j][2]
#             print("error", (out[i][j][1]-orig_map[i][j][1])**2)
#             print("error", (out[i][j][2]-orig_map[i][j][2])**2)
#             cv2.rectangle(img, np.int0(((j+0.5)*224/5-(w/2*224), (i+0.5)*224/5-(h/2*224))),np.int0(((j+0.5)*224/5+(w/2*224), (i+0.5)*224/5+(h/2*224))), (255, 0, 0),2)
for i in range(5):
    for j in range(5):
        if out[i][j][0] > 0.5*out.max():
            w = out[i][j][1]
            h = out[i][j][2]
            print(np.int0((j*224/5-(w/2*224), i*224/5-(h/2*224))),np.int0((j*224/5+(w/2*224), i*224/5+(h/2*224))))
            cv2.rectangle(img, np.int0(((j+0.5)*224/5-(w/2*224), (i+0.5)*224/5-(h/2*224))),np.int0(((j+0.5)*224/5+(w/2*224), (i+0.5)*224/5+(h/2*224))), (0, 0, 255),4)
# print(out.min(), out.max())


import matplotlib.pyplot as plt
plt.imshow(img)
plt.figure()
plt.imshow(out[:,:,1])
plt.show()

# img = cv2.imread("dataset/Preprocessed/002675.png")
# out = model(img[None,:,:,:]/255.).numpy()[0]
# out = np.clip(out, 0, 1)
# print(out[:,:,0])
# for i in range(5):
#     for j in range(5):
#         if out[i][j][0] > 0.5*out.max():
#             w = out[i][j][1]
#             h = out[i][j][2]
#             # print(np.int0(((j+0.5)*224/5-(w/2*224), i*224/5-(h/2*224/5))),np.int0((j*224/5+(w/2*224/5), i*224/5+(h/2*224/5))))
#             cv2.rectangle(img, np.int0(((j+0.5)*224/5-(w/2*224), (i+0.5)*224/5-(h/2*224))),np.int0(((j+0.5)*224/5+(w/2*224), (i+0.5)*224/5+(h/2*224))), (0, 0, 255),2)
# print(out.min(), out.max())

# import matplotlib.pyplot as plt
# plt.imshow(img)
# plt.figure()
# plt.imshow(out[:,:,0])
# plt.show()

import cv2
# orig_map = orig_map.numpy()
img = cv2.imread(X_test[2])#"dataset/Preprocessed/000019.png")#df.image.tolist()[0])
# img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
out = model(img[None,:,:,::-1]/255.).numpy()[0]
# print(np.mean(img/255.), img.min())
# exit()
print(out[:,:,0])
# out = np.clip(out, 0, 1)
out = np.clip(out, 0, 1)
print(out[:,:,0])
# for i in range(5):
#     for j in range(5):
#         if orig_map[i][j][0] > 0:
#             w = out[i][j][1]
#             h = out[i][j][2]
#             print("error", (out[i][j][1]-orig_map[i][j][1])**2)
#             print("error", (out[i][j][2]-orig_map[i][j][2])**2)
#             cv2.rectangle(img, np.int0(((j+0.5)*224/5-(w/2*224), (i+0.5)*224/5-(h/2*224))),np.int0(((j+0.5)*224/5+(w/2*224), (i+0.5)*224/5+(h/2*224))), (255, 0, 0),2)
for i in range(5):
    for j in range(5):
        if out[i][j][0] > 0.5*out.max():
            w = out[i][j][1]
            h = out[i][j][2]
            print(np.int0((j*224/5-(w/2*224), i*224/5-(h/2*224))),np.int0((j*224/5+(w/2*224), i*224/5+(h/2*224))))
            cv2.rectangle(img, np.int0(((j+0.5)*224/5-(w/2*224), (i+0.5)*224/5-(h/2*224))),np.int0(((j+0.5)*224/5+(w/2*224), (i+0.5)*224/5+(h/2*224))), (0, 0, 255),4)
# print(out.min(), out.max())


import matplotlib.pyplot as plt
plt.imshow(img)
plt.figure()
plt.imshow(out[:,:,1])
plt.show()

# img = cv2.imread("dataset/Preprocessed/002675.png")
# out = model(img[None,:,:,:]/255.).numpy()[0]
# out = np.clip(out, 0, 1)
# print(out[:,:,0])
# for i in range(5):
#     for j in range(5):
#         if out[i][j][0] > 0.5*out.max():
#             w = out[i][j][1]
#             h = out[i][j][2]
#             # print(np.int0(((j+0.5)*224/5-(w/2*224), i*224/5-(h/2*224/5))),np.int0((j*224/5+(w/2*224/5), i*224/5+(h/2*224/5))))
#             cv2.rectangle(img, np.int0(((j+0.5)*224/5-(w/2*224), (i+0.5)*224/5-(h/2*224))),np.int0(((j+0.5)*224/5+(w/2*224), (i+0.5)*224/5+(h/2*224))), (0, 0, 255),2)
# print(out.min(), out.max())

# import matplotlib.pyplot as plt
# plt.imshow(img)
# plt.figure()
# plt.imshow(out[:,:,0])
# plt.show()