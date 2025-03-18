import numpy as np
import numpy as np
import yfinance as yf
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import datetime
import tensorflow as tf
import django

model = tf.keras.models.load_model("models/tai_0.6187574516006298.h5")
model2 = tf.keras.models.load_model("models/tai_0.6176295657813878.h5")
model3 = tf.keras.models.load_model("models/tai_0.6845681798314436.h5")



stocks = []
data = np.genfromtxt('all_stocks_5yr.csv', delimiter=',', dtype=None, encoding=None)
s = data[0][6]
stocks.append(s)
for i in data:
    if i[6] != s:
        s = i[6]
        stocks.append(i[6])

def date_to_num(date):
    m = date[5:7]
    d = date[8:10]
    return int(m) / 100, int(d) / 100

def get_data(s, y, m, d):
    days = 120
    start = datetime.date(y, m, d) - datetime.timedelta(days=days)
    data = yf.download(s, start, datetime.date(y, m, d),  progress=False)
    while len(data) < 90:
        days += 1
        start = datetime.date(y, m, d) - datetime.timedelta(days=days)
        data = yf.download(s, start, datetime.date(y, m, d), progress=False)

    dates = data.index
    dates_l = []

    for i in dates:
        dates_l.append(str(i).replace(" 00:00:00", ""))
    data = data.values.tolist()
    print(data[-1][3])

    # real_price = data[-1][3]
    # del data[-1]
    # answer = ((real_price - data[-1][3]) / data[-1][3]) * 100
    sum_data = []
    for i in range(len(data)):
        data[i].pop(4)
        data[i][0] /= 1000000
        data[i][1] /= 1000000
        data[i][2] /= 1000000
        data[i][3] /= 1000000
        data[i][4] /= 100000000000
        data[i].insert(0, date_to_num(dates_l[i])[1])
        data[i].insert(0, date_to_num(dates_l[i])[0])

        sum_data.append(data[i][0])
        sum_data.append(data[i][1])
        sum_data.append(data[i][2])
        sum_data.append(data[i][3])
        sum_data.append(data[i][4])
        sum_data.append(data[i][5])
        sum_data.append(data[i][6])
    p1 = model.predict([sum_data])[0]
    p2 = model2.predict([sum_data])[0]
    p3 = model3.predict([sum_data])[0]
    p = [(p1[0] + p2[0] + p3[0]) / 3, (p1[1] + p2[1] + p3[1]) / 3]
    return np.argmax(p), np.max(p)


stocks = [
    "AAPL",
    "NVDA",
    "MSFT",
    "GOOGL",
    "AMZN",
    "META",
    "TSLA"]


y = 2024
m = 10
d = 4

up = []
down = []
c = 7
for i in stocks:
    print(c)
    c -= 1
    if(get_data(i, y, m, d)[0]) == 0:
        up.append([i, get_data(i, y, m, d)[1]])
    else:
        down.append([i, get_data(i, y, m, d)[1]])


up.sort(reverse=True, key=lambda t: t[1])
down.sort(reverse=True, key=lambda t: t[1])


print("up: ")
print()
for i in up:
    print(f"{i[0]} - acc: {i[1]}")
print()
print("=======================")
print()
print("down: ")
print()
for i in down:
    print(f"{i[0]} - acc: {i[1]}")
















