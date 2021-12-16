import numpy as np
from sys import argv


def sigmoid(x): return 1/(1+np.exp(-x))


def log():
    msg = ''
    for item in test_y:
        msg += f"{item}\n"
    with open('test_y', 'w') as out:
        out.write(msg)


def fprop(x):
    global w1, w2, b1, b2, z1, z2
    z1 = (np.dot(w1, x) + b1)/128
    h1 = sigmoid(z1)
    z2 = (np.dot(w2, h1) + b2)/10
    return h1, softmax(z2)


def bprop(x, y, h1, h2):
    global w1, w2, b1, b2, z1, z2
    y_hat = np.zeros((10, 1))
    y_hat[y] = 1
    dz2 = h2 - y_hat
    dw2, dz1 = np.dot(dz2, h1.T), np.dot(w2.T, dz2)*sigmoid(z1)*(1-sigmoid(z1))
    dw1 = np.dot(dz1, x.T)
    return dw1, dz1, dw2, dz2


def softmax(arr):
    arr -= np.max(arr)
    exps = np.exp(arr)
    return exps / exps.sum()


def train():
    global w1, w2, b1, b2
    ETA, EPOCHS = 0.01, 20
    for i in range(EPOCHS):
        print(f'Running epoch {i+1}/{EPOCHS}')
        s = np.random.permutation(train_size)
        train_set = zip(train_x[s], train_y[s])
        for x, y in train_set:
            x = x.reshape((784, 1))
            h1, h2 = fprop(x)
            dw1, dz1, dw2, dz2 = bprop(x, y, h1, h2)
            b1 -= dz1*ETA
            w1 -= dw1*ETA
            b2 -= dz2*ETA
            w2 -= dw2*ETA


def test():
    for x in test_x:
        _, h2 = fprop(x.reshape((784, 1)))
        test_y.append(np.argmax(h2))


train_x = np.genfromtxt(argv[1]) / 255
train_y, test_y = np.genfromtxt(argv[2]).astype(int), []
test_x = np.genfromtxt(argv[3])/255
train_size, test_size = train_x.shape[0], test_x.shape[0]
w1, w2 = np.random.rand(128, 784), np.random.rand(10, 128)
b1, b2, z1, z2 = np.random.rand(128, 1), np.random.rand(10, 1), 0, 0
train(), test(), log()
