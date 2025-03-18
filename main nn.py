import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from main_2 import train
models = []
# model = tf.keras.Sequential([
#     tf.keras.layers.Dense(256, activation='relu', input_shape=(630,)),
#     tf.keras.layers.Dropout(0.3),
#
#     tf.keras.layers.Dense(128, activation='relu'),
#     tf.keras.layers.BatchNormalization(),
#
#     tf.keras.layers.Dense(100, activation='relu'),
#     tf.keras.layers.Dropout(0.2),
#
#     tf.keras.layers.Dense(80, activation='relu'),
#     tf.keras.layers.BatchNormalization(),
#
#     tf.keras.layers.Dense(64, activation='relu'),
#     tf.keras.layers.Dropout(0.1),
#
#     tf.keras.layers.Dense(32, activation='relu'),
#     tf.keras.layers.BatchNormalization(),
#
#     tf.keras.layers.Dense(16, activation='relu'),
#     tf.keras.layers.BatchNormalization(),
#
#     tf.keras.layers.Dense(2, activation='softmax')  # Assuming binary classification with 2 classes
# ])
model = tf.keras.Sequential([
    tf.keras.layers.Dense(540, activation='elu', input_shape=(630,)),  # Hidden layer with ReLU
    tf.keras.layers.Dropout(0.3),

    tf.keras.layers.Dense(256, activation='elu'),  # Hidden layer with ReLU
    tf.keras.layers.BatchNormalization(),

    tf.keras.layers.Dense(100, activation='elu'),  # Hidden layer with ReLU
    tf.keras.layers.Dropout(0.2),

    tf.keras.layers.Dense(46, activation='elu'),  # Hidden layer with ReLU
    tf.keras.layers.BatchNormalization(),

    tf.keras.layers.Dense(26, activation='elu'),  # Hidden layer with ReLU
    tf.keras.layers.Dense(8, activation='elu'),  # Hidden layer with ReLU
    tf.keras.layers.Dense(2, activation='softmax')  # Output layer with Softmax for multi-class classification
])






# model = tf.keras.models.load_model("models/tai_0.6150233319973819.h5")
lr = 0.000001
runs = 2
acc = 0
ephocs = 500
batch_size = 8
new_model = model
for i in range(runs):
    print(f"{i+1}/{runs}. lr: {lr}  last acc: {acc}")
    new_model, acc = train(new_model, lr, ephocs, batch_size)
    models.append([i, lr, acc])
    lr = lr / 10

n = []

for i in models:
    n.append(i[2])
    print(f"{i[0]}. lr: {i[1]}  acc: {i[2]}")

plt.plot(n)
plt.show()















