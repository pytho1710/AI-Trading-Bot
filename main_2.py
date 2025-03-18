import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
name = "8"

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

def train(model, lr, epochs, batch_size):
    X = np.load('data/X_1_or_0_with_mul.npy')
    Y = np.load('data/Y_1_or_0_with_mul.npy')


    x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=42)
    print("=================================")
    print("starting training...")


    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=lr),
                  loss=tf.keras.losses.CategoricalCrossentropy(),
                  metrics=[tf.keras.metrics.CategoricalAccuracy()])

    loss_values = []
    epoch_values = []

    loss_plot_callback = LossPlotCallback(epoch_values, loss_values)

    model.fit(x_train, y_train, epochs=epochs, batch_size=batch_size,
              validation_data=(x_test, y_test), callbacks=[loss_plot_callback])


    preds = model.predict(x_test)
    loss = preds - y_test
    a = 0
    for i in loss:
        a += np.max(i)
    acc = a / len(loss)

    print(f'\nTest accuracy: {1 - acc}')
    model.save(f"models/tai_{1 - acc}.h5")
    plt.close()
    return model, 1 - acc
