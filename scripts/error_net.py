import tensorflow as tf
from tensorflow import keras
import mdn
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scripts.utils import generate_self_vector
from scripts.utils import generate_self_vector, load_data

#
#
# x_train = np.load("../../data/1filter_points.npy")
# y_train = np.load("../../data/1op_position.npy")
#
# fig = plt.figure()
# ax = fig.add_subplot(111, projection='3d')
# ax.scatter(x_train[:,0], x_train[:,1], x_train[:,2], alpha=0.3, c='r')  #c=perf_down_sampled.moving
# ax.scatter(y_train[:,0], y_train[:,1], y_train[:,2]*0.0, alpha=0.3, c='b')
# plt.show()
#
# plt.scatter(x_train[:,0], x_train[:,1])
# plt.scatter(y_train[:,0], y_train[:,1])
# plt.show()
#
#
#
#
# q = np.load("../../data/q_sim.npy")
# x_train = np.reshape(x_train, (-1, 24, 3))
# x_train = np.reshape(x_train, (-1, 72,))
# y_train = np.load("../../data/human_position_sim.npy")
# y_train = y_train[:, :-1]
# x_train_new = []
# y_train_new = []
# qs = []
#
# for idx, each in enumerate(x_train):
#     if np.sum(each) != 0:
#         x_train_new.append(each)
#         y_train_new.append(y_train[idx])
#         qs.append(q[idx])
# x_train_new = np.vstack(x_train_new)
# qs = np.vstack(qs)
# x_train_new = np.hstack([x_train_new, qs])
#
# print(q.shape)
# y_train_new = np.vstack(y_train_new)
#
#
#
#
#
#
#
#
# #(v - v.min()) / (v.max() - v.min())
# #x_train_new = (x_train_new - np.mean(x_train_new, axis=0))/(np.std(x_train_new, axis=0))# - np.min(x_train_new, axis=0))
#
# # print(y_train.shape)
# x_test = np.load("../../data/pred_pc_real.npy")
#
#
#
# q_test = np.load("../../data/q_real.npy")
# x_test = np.reshape(x_test, (-1, 24, 3))
# x_test = np.reshape(x_test, (-1, 72,))
# y_test = np.load("../../data/human_position_real.npy")
# y_test = y_test[:, :-1]
# x_test_new = []
# y_test_new = []
# qr = []
# for idx, each in enumerate(x_test):
#     if np.sum(each) != 0:
#         x_test_new.append(each)
#         y_test_new.append(y_test[idx])
#         qr.append(q_test[idx])
#
# x_test_new = np.vstack(x_test_new)
# qr = np.vstack(qr)
# x_test_new = np.hstack([x_test_new, qr])
# y_test_new = np.vstack(y_test_new)
def rmse_loss(targets, outputs):
    return tf.sqrt(tf.reduce_mean(tf.square(tf.subtract(outputs, targets))))


pc = "../data/pc_with_object_ids13.npy"
hp = "../data/human_pos13.npy"
q = "../data/q13.npy"

pct = "../data/pc_with_object_ids12.npy"
hpt = "../data/human_pos12.npy"
qt = "../data/q12.npy"


x_train, y_train, x_train_ = load_data(pc, hp, q)
# #
# #
# # x_train = np.load("../../data/1filter_points.npy")
# # y_train = np.load("../../data/1op_position.npy")
# #
# # fig = plt.figure()
# # ax = fig.add_subplot(111, projection='3d')
# # ax.scatter(x_train[:,0], x_train[:,1], x_train[:,2], alpha=0.3, c='r')  #c=perf_down_sampled.moving
# # ax.scatter(y_train[:,0], y_train[:,1], y_train[:,2]*0.0, alpha=0.3, c='b')
# # plt.show()
# #
# # plt.scatter(x_train[:,0], x_train[:,1])
# # plt.scatter(y_train[:,0], y_train[:,1])
# # plt.show()
# #
# #
# #
# #
# # q = np.load("../../data/q_sim.npy")
# # x_train = np.reshape(x_train, (-1, 24, 3))
# # x_train = np.reshape(x_train, (-1, 72,))
# # y_train = np.load("../../data/human_position_sim.npy")
# # y_train = y_train[:, :-1]
# # x_train_new = []
# # y_train_new = []
# # qs = []
# #
# # for idx, each in enumerate(x_train):
# #     if np.sum(each) != 0:
# #         x_train_new.append(each)
# #         y_train_new.append(y_train[idx])
# #         qs.append(q[idx])
# # x_train_new = np.vstack(x_train_new)
# # qs = np.vstack(qs)
# # x_train_new = np.hstack([x_train_new, qs])
# #
# # print(q.shape)
# # y_train_new = np.vstack(y_train_new)
# #
# #
# #
# #
# #
# #
# #
# #
# # #(v - v.min()) / (v.max() - v.min())
# # #x_train_new = (x_train_new - np.mean(x_train_new, axis=0))/(np.std(x_train_new, axis=0))# - np.min(x_train_new, axis=0))
# #
# # # print(y_train.shape)
# # x_test = np.load("../../data/pred_pc_real.npy")
# #
# #
# #
# # q_test = np.load("../../data/q_real.npy")
# # x_test = np.reshape(x_test, (-1, 24, 3))
# # x_test = np.reshape(x_test, (-1, 72,))
# # y_test = np.load("../../data/human_position_real.npy")
# # y_test = y_test[:, :-1]
# # x_test_new = []
# # y_test_new = []
# # qr = []
# # for idx, each in enumerate(x_test):
# #     if np.sum(each) != 0:
# #         x_test_new.append(each)
# #         y_test_new.append(y_test[idx])
# #         qr.append(q_test[idx])
# #
# # x_test_new = np.vstack(x_test_new)
# # qr = np.vstack(qr)
# # x_test_new = np.hstack([x_test_new, qr])
# # y_test_new = np.vstack(y_test_new)
# def rmse_loss(targets, outputs):
#     return tf.sqrt(tf.reduce_mean(tf.square(tf.subtract(targets, outputs))))
#
#
# x_train = np.load("../data/pc_with_object_ids12.npy")
# y_train = np.load("../data/human_pos12.npy")
# q = np.load(("../data/q12.npy"))
# x_train_ = []
#
# for point_cloud in range(len(x_train)):
#     for point in range(24):
#         if x_train[point_cloud][point][-1] == 0:
#             x_train[point_cloud][point][:-1] = np.zeros((3,))
# # x_train = np.reshape(x_train[:, :, :-1], (-1, 3))
# # print(x_train.shape)
# #  ZERO POINT CLOUD REMOVAL
# x_filtered = []
# y_filtered = []
# q_filtered = []
# for idx, each in enumerate(x_train[:, :, :-1]):
#     if np.sum(each) != 0:
#         x_filtered.append(each)
#         y_filtered.append(y_train[idx].flatten())
#         q_filtered.append(q[idx])
# x_filtered = np.vstack(x_filtered)
# print(x_filtered.shape)
# x_filtered = np.reshape(x_filtered, (-1, 24, 3))
# x_filtered = np.reshape(x_filtered, (-1, 72))
# y_filtered = np.vstack(y_filtered)
# q_filtered = np.vstack(q_filtered)
#
# #plt.scatter(y_filtered[:, 0], y_filtered[:, 1])
# #plt.scatter(x_filtered[:, 0], x_filtered[:, 1])
#
# #plt.show()
#
# #x_train = np.hstack([x_filtered, q_filtered])
# x_train = x_filtered
# x_train_ = x_train
# x_train = np.hstack([x_filtered, q_filtered])
# #x_train = np.reshape(x_train, (-1, 78))
# y_train = y_filtered[:, :-1]
#
# x_shuff = np.hstack([x_train, y_train])
# np.random.shuffle(x_shuff)
#
# x_train = np.vstack([x_train, x_shuff[:, :-2]])
# y_train = np.vstack([y_train, x_shuff[:, -2:]])
#
# x_shuff = np.hstack([x_train, y_train])
# np.random.shuffle(x_shuff)
#
# x_train = np.vstack([x_train, x_shuff[:, :-2]])
# y_train = np.vstack([y_train, x_shuff[:, -2:]])
#
# x_train__ = np.reshape(x_train[:, :-6], (-1,24,3))
# x_train__ = x_train__ + np.random.randn(x_train__.shape[0], 24, 3)*0.1
# x_train[:, :-6] = np.reshape(x_train__, (-1, 72))

raw_x = np.reshape(x_train[:, :-6], (-1, 24, 3))
#raw_x_1 = np.reshape(raw_x, (-1, 24, 3))
centroids = []

for each in raw_x:
    non_zero = each[~np.all(each == 0, axis=1)]
    centroids.append(np.mean(non_zero, axis=0))
centroids = np.vstack(centroids)
print(centroids.shape)
print(y_train.shape)

errors = np.var([centroids[:, :-1], y_train], axis=0)

x_train = centroids[:, :-1]
y_train = errors

N_MIXES = 1
OUTPUT_DIMS = 2
sgd = keras.optimizers.SGD(lr=0.0001, decay=1e-6, momentum=0.9, nesterov=True)
adam = keras.optimizers.Adam()
model = keras.Sequential()
model.add(keras.layers.Dense(8, batch_input_shape=(None, 2), activation='relu'))
model.add(keras.layers.Dropout(0.2))
model.add(keras.layers.Dense(64, activation='relu'))
model.add(keras.layers.Dropout(0.2))
model.add(keras.layers.Dense(64, activation='tanh'))
model.add(keras.layers.Dropout(0.1))
model.add(mdn.MDN(OUTPUT_DIMS, N_MIXES))
model.compile(loss=mdn.get_mixture_loss_func(OUTPUT_DIMS, N_MIXES), optimizer=sgd)
model.summary()

es = keras.callbacks.EarlyStopping(monitor='val_loss', min_delta=0, patience=10, verbose=0, mode='auto', baseline=None, restore_best_weights=False)
ckpt = keras.callbacks.ModelCheckpoint("enet.h5", monitor='val_loss', verbose=0, save_best_only=True, save_weights_only=False, mode='auto', period=1)
history = model.fit(x=x_train, y=y_train, batch_size=256, epochs=200, validation_split=0.10, callbacks=[keras.callbacks.TerminateOnNaN(), es, ckpt])
#
plt.figure(figsize=(10, 5))
plt.ylim([0, 9])
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.show()

model.load_weights("enet.h5")


x_test, y_test, x_test_ = load_data(pct, hpt, qt)

centroids_ = []
raw_x_ = np.reshape(x_test[:, :-6], (-1, 24, 3))
for each in raw_x_:
    non_zero = each[~np.all(each == 0, axis=1)]
    centroids_.append(np.mean(non_zero, axis=0))
centroids_ = np.vstack(centroids_)
print(centroids_.shape)
print(y_test.shape)

errors_t = np.var([centroids_[:, :-1], y_test], axis=0)

x_test = centroids_[:, :-1]
y_test = errors_t

y_pred = model.predict(x_test)

y_samples = np.apply_along_axis(mdn.sample_from_output, 1, y_pred, OUTPUT_DIMS, N_MIXES, temp=1.5, sigma_temp=0)
y_samples = np.reshape(y_samples, (-1, 2))
print(rmse_loss(y_test, y_samples).numpy())




plt.scatter(y_samples[:, 0], y_samples[:, 1])
plt.scatter(y_test[:, 0], y_test[:, 1])
plt.scatter(x_test[:, 0], x_test[:, 1])
plt.show()


