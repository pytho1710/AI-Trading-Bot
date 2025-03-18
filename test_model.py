import numpy as np
import yfinance as yf
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import datetime
import tensorflow as tf

model = tf.keras.models.load_model("models/tai_0.6120301279605136.h5")

X = []
Y = []
def date_to_num(date):
    m = date[5:7]
    d = date[8:10]
    return int(m) / 100, int(d) / 100
def get_data(s, y, m, d):
    days = 120
    start = datetime.date(y, m, d) - datetime.timedelta(days=days)
    data = yf.download(s, start, datetime.date(y, m, d),  progress=False)
    while len(data) < 91:
        days += 1
        start = datetime.date(y, m, d) - datetime.timedelta(days=days)
        data = yf.download(s, start, datetime.date(y, m, d), progress=False)
    dates = data.index
    dates_l = []

    for i in dates:
        dates_l.append(str(i).replace(" 00:00:00", ""))
    data = data.values.tolist()
    y = data[-1][3]
    data.pop(-1)
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
    p = model.predict([sum_data])
    return (p[0] * 1000000)[0], y


def get_real_prices(s, y, m, d, days):
    static_num_ofdays = days
    days = days + ((days / 7) * 3)
    start = datetime.date(y, m, d) - datetime.timedelta(days=days)
    data = yf.download(s, start, datetime.date(y, m, d),  progress=False)
    while len(data) < static_num_ofdays:
        days += 1
        start = datetime.date(y, m, d) - datetime.timedelta(days=days)
        data = yf.download(s, start, datetime.date(y, m, d), progress=True)
    return data["Close"].values.tolist()


def get_pre_data(s, y, m, d, days):
    data = []
    d = datetime.date(y, m, d)

    for i in range(days):
        d = d + datetime.timedelta(1)
        if d.weekday() == 6:
            d = d + datetime.timedelta(1)
        elif d.weekday() == 5:
            d = d + datetime.timedelta(2)
        data.append(get_data(s, d.year, d.month, d.day)[0])
    print(f"pre: {data}")
    return data


