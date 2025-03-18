import numpy as np

data = np.genfromtxt('all_stocks_5yr.csv', delimiter=',', dtype=None, encoding=None)
data_to_lists_by_stocks = []
s = data[1][6]
f = False
id = 1
data = data.tolist()
del data[0]

def mul(n):
    while n > 0.1:
        n /= 10
    return n * 10


for i in range(len(data)):
    if i % 1259 == 0:
        data_to_lists_by_stocks.append(data[i-1259:i])
del data_to_lists_by_stocks[0]


def list_of_dates(dates):
    m = []
    d = []
    for i in dates:
        print(i)
        m.append(date_to_num(i)[0])
        d.append(date_to_num(i)[1])
    return m, d
def date_to_num(date):
    date = date.split("/")
    m = date[0]
    d = date[1]
    return int(m) / 100, int(d) / 100

X = []
Y = []
c = 0


for stock in data_to_lists_by_stocks:
    print(c)
    c += 1
    for i in range(0, 1168):
        x = []
        can_append = True
        for j in stock[i:i+90]:
            x.append(date_to_num(j[0])[0])
            x.append(date_to_num(j[0])[1])
            x.append(mul(float(j[1])))
            x.append(mul(float(j[2])))
            x.append(mul(float(j[3])))
            x.append(mul(float(j[4])))
            x.append(mul(float(j[5])))
        X.append(x),
        if float(stock[i+91][4]) > float(stock[i+90][4]):
            Y.append([1, 0])
        else:
            Y.append([0, 1])


print(Y)
print(len(X))
print(len(Y))
np.save(f"data/X_1_or_0_with_mul.npy", np.array(X))
np.save(f"data/Y_1_or_0_with_mul.npy", np.array(Y))










