import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt


name = "16"

class LossPlotCallback(tf.keras.callbacks.Callback):
    def __init__(self, x, y):
        self.x = x
        self.y = y
        self.fig, self.ax = plt.subplots()
        self.line, = self.ax.plot(x, y, 'b-', label=name)
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.title(name)

    def on_epoch_end(self, epoch, logs=None):
        loss = logs.get('loss')
        self.x.append(epoch)
        self.y.append(loss)
        self.line.set_data(self.x, self.y)
        self.ax.relim()
        self.ax.autoscale_view()
        plt.pause(0.01)

X = np.load('data/X1_or_0.npy')
Y = np.load('data/Y1_or_0.npy')
print(len(X))
print(len(Y))
print(X[0])
print(Y[0])



x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=42)
print("=================================")
print(x_test.shape)
print(y_test.shape)
print(y_test[:10])
n = 640
nu = []
for i in range(30):
    nu.append(n)
    n -= 80
# model = tf.keras.Sequential([
    # tf.keras.layers.Dense(nu[0], activation='relu', input_shape=(630,)),
    # tf.keras.layers.Dense(nu[1], activation='relu'),
    # tf.keras.layers.Dense(nu[2], activation='relu'),
    # tf.keras.layers.Dropout(0.3),
    # tf.keras.layers.Dense(nu[3], activation='relu'),
    # tf.keras.layers.Dense(nu[4], activation='relu'),
    # tf.keras.layers.Dense(2, activation='softmax')
# ])
model = tf.keras.models.load_model("models/tai_0.6133351094992181.h5")

learning_rate = 0.00000000001
epochs = 100
batch_size = 24

model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
              loss=tf.keras.losses.CategoricalCrossentropy(),
              metrics=[tf.keras.metrics.CategoricalAccuracy()])


loss_values = []
epoch_values = []

loss_plot_callback = LossPlotCallback(epoch_values, loss_values)

model.fit(x_train, y_train, epochs=epochs, batch_size=batch_size,
          validation_data=(x_test, y_test), callbacks=[loss_plot_callback])



preds = model.predict(x_test)
print(preds - y_test)
loss = preds - y_test
a = 0
for i in loss:
    a += np.max(i)
acc = a / len(loss)

print(f'\nTest accuracy: {1 - acc}')
model.save(f"models/tai_{1 - acc}.h5")

