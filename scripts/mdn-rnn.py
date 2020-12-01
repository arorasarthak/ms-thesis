import tensorflow as tf
from tensorflow import keras

import mdn
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import random
from mpl_toolkits.mplot3d import Axes3D
from scripts.utils import generate_self_vector, load_data

from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession
config = ConfigProto()
config.gpu_options.allow_growth = True
session = InteractiveSession(config=config)


pc = "../data/pc_with_object_ids13.npy"
hp = "../data/human_pos13.npy"
q = "../data/q13.npy"

pct = "../../data/pc_with_object_ids14.npy"
hpt = "../../data/human_pos14.npy"
qt = "../../data/q14.npy"


x_train, y_train, x_train_ = load_data(pc, hp, q)

SEQ_LEN = 10
BATCH_SIZE = 256
HIDDEN_UNITS = 256
EPOCHS = 100
VAL_SPLIT = 0.15

# Set random seed for reproducibility
SEED = 2345
random.seed(SEED)
np.random.seed(SEED)

def slice_sequence_examples(sequence, num_steps):
    xs = []
    for i in range(len(sequence) - num_steps - 1):
        example = sequence[i: i + num_steps]
        xs.append(example)
    return xs

def seq_to_singleton_format(examples):
    xs = []
    ys = []
    for ex in examples:
        xs.append(ex[:-1])
        ys.append(ex[-1])
    return (xs,ys)


sequences = slice_sequence_examples(y_train, SEQ_LEN+1)
print("Total training examples:", len(sequences))
X, y = seq_to_singleton_format(sequences)
X = np.array(X)
y = np.array(y)
print("X:", X.shape, "y:", y.shape)


OUTPUT_DIMENSION = 2
NUMBER_MIXTURES = 5


model = keras.Sequential()
model.add(keras.layers.LSTM(HIDDEN_UNITS, batch_input_shape=(None, SEQ_LEN, OUTPUT_DIMENSION), return_sequences=True))
model.add(keras.layers.LSTM(HIDDEN_UNITS))
model.add(mdn.MDN(OUTPUT_DIMENSION, NUMBER_MIXTURES))
model.compile(loss=mdn.get_mixture_loss_func(OUTPUT_DIMENSION,NUMBER_MIXTURES), optimizer=keras.optimizers.Adam())
model.summary()

# Train the model

# Define callbacks
filepath = "mdn-rnn.h5"
checkpoint = keras.callbacks.ModelCheckpoint(filepath, save_weights_only=True, verbose=1, save_best_only=True, mode='min')
early_stopping = keras.callbacks.EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=10)
callbacks = [keras.callbacks.TerminateOnNaN(), checkpoint, early_stopping]

history = model.fit(X, y, batch_size=BATCH_SIZE, epochs=EPOCHS, callbacks=callbacks, validation_split=VAL_SPLIT)

# Save the Model
model.save('mdn-rnn.h5')  # creates a HDF5 file of the model

# Plot the loss
plt.figure(figsize=(10, 5))
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.xlabel("epochs")
plt.ylabel("loss")
plt.show()