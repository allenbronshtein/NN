import numpy as np
from sys import argv


def log():
    msg = ''
    for item in test_y:
        msg += f"{item}\n"
    with open('test_y', 'w') as out:
        out.write(msg)


def fprop(x):
    global w1, w2, w3, b1, b2, b3
    z1 = np.dot(w1, x) + b1
    h1 = 1 / (1 + np.exp(-1.0 * z1))
    z2 = np.dot(w2, h1) + b2
    h2 = 1 / (1 + np.exp(-1.0 * z2))
    z3 = np.dot(w3, h2) + b3
    return h1, h2, softmax(z3)


def bprop(x, y, h1, h2, h3):
    global w1, w2, w3, b1, b2, b3
    y_hat = np.zeros((10, 1))
    y_hat[y] = 1
    dz3 = h3 - y_hat
    dw3 = np.dot(dz3, np.transpose(h2))
    dz2 = np.dot(np.transpose(w3), dz3) * h2 * (1 - h2)
    dw2 = np.dot(dz2, np.transpose(h1))
    dz1 = np.dot(np.transpose(w2), dz2) * h1 * (1 - h1)
    dw1 = np.dot(dz1, np.transpose(x))
    return dw1, dz1, dw2, dz2, dw3, dz3


def softmax(arr):
    arr -= np.max(arr)
    exps = np.exp(arr)
    return exps / exps.sum()


def train():
    global w1, w2, w3, b1, b2, b3
    ETA, EPOCHS = 0.01, 4
    for _ in range(EPOCHS):
        s = np.random.permutation(train_size)
        train_set = zip(train_x[s], train_y[s])
        for x, y in train_set:
            x = x.reshape((784, 1))
            h1, h2, h3 = fprop(x)
            dw1, dz1, dw2, dz2, dw3, dz3 = bprop(x, y, h1, h2, h3)
            b1 -= dz1*ETA
            w1 -= dw1*ETA
            b2 -= dz2*ETA
            w2 -= dw2*ETA
            b3 -= dz3*ETA
            w3 -= dw3*ETA
    print("training done")


def test():
    for x in test_x:
        _, _, h3 = fprop(x.reshape((784, 1)))
        test_y.append(np.argmax(h3))


train_x = np.genfromtxt('NN/train_x') / 255
test_x = np.genfromtxt('NN/test_x')/255
train_y, test_y = np.genfromtxt('NN/train_y').astype(int), []
train_size, test_size = train_x.shape[0], test_x.shape[0]
w1, w2, w3 = np.random.rand(250, 784), np.random.rand(
    100, 250), np.random.rand(10, 100)
b1, b2, b3 = np.random.rand(250, 1), np.random.rand(
    100, 1), np.random.rand(10, 1)
train(), test(), log()
