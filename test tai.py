import numpy as np
import numpy as np
import yfinance as yf
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import datetime
import tensorflow as tf

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
    days = 127
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

    real_price = data[-1][3]
    del data[-1]
    # answer = ((real_price - data[-1][3]) / data[-1][3]) * 100
    answer = 100 * (1 + (((real_price - data[-1][3]) / data[-1][3]) * 100) / 100)
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
    return np.argmax(p), np.max(p), answer - 100


stocks = [
    "AAPL",
    "NVDA",
    "MSFT",
    "GOOGL",
    "AMZN",
    "META",
    "TSLA"]

def profit_to_date(y, m, d):
    up = []
    down = []
    c = 7
    for i in stocks:
        c -= 1
        if (get_data(i, y, m, d)[0]) == 0:
            up.append([i, get_data(i, y, m, d)[1], get_data(i, y, m, d)[2]])
        else:
            down.append([i, get_data(i, y, m, d)[1], get_data(i, y, m, d)[2]])

    sum = 0
    for i in up:
        sum += i[2]
        print(i)
    print(f"you startes with {len(up) * 100} and ended up with {len(up) * 100 + sum}")
    before = len(up) * 100
    after = len(up) * 100 + sum
    if before != 0:
        print(f"profit: {sum}  |  {((after - before) / before) * 100}%")
    return sum

profit = []


date = datetime.datetime(2014, 1, 1)
# y = 2024
# m = 8
# d = 1
s = 0

days = 3500

for i in range(days):
    print(days - i)
    date += datetime.timedelta(days=1)
    s += profit_to_date(date.year, date.month, date.day)
    profit.append(s)
print()
print()
print("=============")
print()
print()
print(f"over all profit: {round(s, 3)}  |  {((700 + s - 700) / 700) * 100}%")

plt.plot(profit)
plt.show()















