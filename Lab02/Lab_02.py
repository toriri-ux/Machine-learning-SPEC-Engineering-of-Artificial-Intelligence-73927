import numpy as np
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
import time
import matplotlib.pyplot as plt

digits = load_digits()
X = digits.data / 16.0
y = digits.target.reshape(-1, 1)

enc = OneHotEncoder(sparse_output=False)
Y = enc.fit_transform(y)

X_train, X_test, Y_train, Y_test = train_test_split(
    X, Y, test_size=0.3, stratify=y, random_state=42
)


def relu(x):
    return np.maximum(0, x)

def relu_grad(x):
    return (x > 0).astype(float)

def softmax(z):
    z -= np.max(z, axis=1, keepdims=True)
    exp = np.exp(z)
    return exp / np.sum(exp, axis=1, keepdims=True)

def cross_entropy(y_hat, y):
    return -np.mean(np.sum(y * np.log(y_hat + 1e-12), axis=1))



class NN:
    def __init__(self, W):
        self.W1 = np.random.randn(64, W) * 0.01
        self.b1 = np.zeros((W,))
        self.W2 = np.random.randn(W, W) * 0.01
        self.b2 = np.zeros((W,))
        self.W3 = np.random.randn(W, 10) * 0.01
        self.b3 = np.zeros((10,))

    def forward(self, X):
        self.z1 = X @ self.W1 + self.b1
        self.a1 = relu(self.z1)

        self.z2 = self.a1 @ self.W2 + self.b2
        self.a2 = relu(self.z2)

        self.z3 = self.a2 @ self.W3 + self.b3
        self.y_hat = softmax(self.z3)
        return self.y_hat

    def backward(self, X, Y):
        m = X.shape[0]

        dz3 = self.y_hat - Y
        self.dW3 = (self.a2.T @ dz3) / m
        self.db3 = np.mean(dz3, axis=0)

        da2 = dz3 @ self.W3.T
        dz2 = da2 * relu_grad(self.z2)
        self.dW2 = (self.a1.T @ dz2) / m
        self.db2 = np.mean(dz2, axis=0)

        da1 = dz2 @ self.W2.T
        dz1 = da1 * relu_grad(self.z1)
        self.dW1 = (X.T @ dz1) / m
        self.db1 = np.mean(dz1, axis=0)

    def step(self, lr):
        for W, dW in [(self.W1, self.dW1), (self.W2, self.dW2), (self.W3, self.dW3)]:
            W -= lr * dW
        for b, db in [(self.b1, self.db1), (self.b2, self.db2), (self.b3, self.db3)]:
            b -= lr * db


def numerical_grad_check(model, X, Y):
    model.forward(X)
    model.backward(X, Y)

    params = [
        ("W1", model.W1, model.dW1),
        ("W2", model.W2, model.dW2),
        ("W3", model.W3, model.dW3)
    ]

    for name, W, dW in params:
        for _ in range(10):
            i = np.random.randint(W.size)
            old = W.flat[i]

            W.flat[i] = old + 1e-5
            loss_plus = cross_entropy(model.forward(X), Y)

            W.flat[i] = old - 1e-5
            loss_minus = cross_entropy(model.forward(X), Y)

            W.flat[i] = old

            num_grad = (loss_plus - loss_minus) / (2e-5)
            ana_grad = dW.flat[i]

            rel_err = abs(num_grad - ana_grad) / (abs(num_grad) + abs(ana_grad) + 1e-12)
            print(name, "relative error:", rel_err)


def train(W, lr):
    model = NN(W)
    batch = 32
    losses = []

    for epoch in range(50):
        idx = np.random.permutation(len(X_train))
        Xb = X_train[idx]
        Yb = Y_train[idx]

        for i in range(0, len(X_train), batch):
            X_batch = Xb[i:i+batch]
            Y_batch = Yb[i:i+batch]

            model.forward(X_batch)
            model.backward(X_batch, Y_batch)
            model.step(lr)

        y_hat = model.forward(X_train)
        loss = cross_entropy(y_hat, Y_train)
        losses.append(loss)
        print(f"Epoch {epoch+1}: loss={loss:.4f}")

    test_pred = np.argmax(model.forward(X_test), axis=1)
    test_true = np.argmax(Y_test, axis=1)
    acc = np.mean(test_pred == test_true)

    return losses, acc


def timing(model, X):
    f_times = []
    b_times = []

    for _ in range(100):
        t = time.time()
        model.forward(X)
        f_times.append(time.time() - t)

        t = time.time()
        model.backward(X, Y_train[:X.shape[0]])
        b_times.append(time.time() - t)

    return np.median(f_times), np.median(b_times)

loss_dict = {}
acc_dict = {}

for W in [16, 64]:
    for lr in [0.001, 0.01, 0.1]:
        losses, acc = train(W, lr)
        loss_dict[(W, lr)] = losses
        acc_dict[(W, lr)] = acc

plt.figure(figsize=(10,6))
for key, curve in loss_dict.items():
    W, lr = key
    plt.plot(curve, label=f"W={W}, lr={lr}")
plt.legend()
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Loss Curves")
plt.savefig("loss_curves.png")

import numpy as np
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
import time
import matplotlib.pyplot as plt

digits = load_digits()
X = digits.data / 16.0
y = digits.target.reshape(-1, 1)

enc = OneHotEncoder(sparse_output=False)
Y = enc.fit_transform(y)

X_train, X_test, Y_train, Y_test = train_test_split(
    X, Y, test_size=0.3, stratify=y, random_state=42
)
 

def relu(x):
    return np.maximum(0, x)

def relu_grad(x):
    return (x > 0).astype(float)

def softmax(z):
    z -= np.max(z, axis=1, keepdims=True)
    exp = np.exp(z)
    return exp / np.sum(exp, axis=1, keepdims=True)

def cross_entropy(y_hat, y):
    return -np.mean(np.sum(y * np.log(y_hat + 1e-12), axis=1))



class NN:
    def __init__(self, W):
        self.W1 = np.random.randn(64, W) * 0.01
        self.b1 = np.zeros((W,))
        self.W2 = np.random.randn(W, W) * 0.01
        self.b2 = np.zeros((W,))
        self.W3 = np.random.randn(W, 10) * 0.01
        self.b3 = np.zeros((10,))

    def forward(self, X):
        self.z1 = X @ self.W1 + self.b1
        self.a1 = relu(self.z1)

        self.z2 = self.a1 @ self.W2 + self.b2
        self.a2 = relu(self.z2)

        self.z3 = self.a2 @ self.W3 + self.b3
        self.y_hat = softmax(self.z3)
        return self.y_hat

    def backward(self, X, Y):
        m = X.shape[0]

        dz3 = self.y_hat - Y
        self.dW3 = (self.a2.T @ dz3) / m
        self.db3 = np.mean(dz3, axis=0)

        da2 = dz3 @ self.W3.T
        dz2 = da2 * relu_grad(self.z2)
        self.dW2 = (self.a1.T @ dz2) / m
        self.db2 = np.mean(dz2, axis=0)

        da1 = dz2 @ self.W2.T
        dz1 = da1 * relu_grad(self.z1)
        self.dW1 = (X.T @ dz1) / m
        self.db1 = np.mean(dz1, axis=0)

    def step(self, lr):
        for W, dW in [(self.W1, self.dW1), (self.W2, self.dW2), (self.W3, self.dW3)]:
            W -= lr * dW
        for b, db in [(self.b1, self.db1), (self.b2, self.db2), (self.b3, self.db3)]:
            b -= lr * db


def numerical_grad_check(model, X, Y):
    model.forward(X)
    model.backward(X, Y)

    params = [
        ("W1", model.W1, model.dW1),
        ("W2", model.W2, model.dW2),
        ("W3", model.W3, model.dW3)
    ]

    for name, W, dW in params:
        for _ in range(10):
            i = np.random.randint(W.size)
            old = W.flat[i]

            W.flat[i] = old + 1e-5
            loss_plus = cross_entropy(model.forward(X), Y)

            W.flat[i] = old - 1e-5
            loss_minus = cross_entropy(model.forward(X), Y)

            W.flat[i] = old

            num_grad = (loss_plus - loss_minus) / (2e-5)
            ana_grad = dW.flat[i]

            rel_err = abs(num_grad - ana_grad) / (abs(num_grad) + abs(ana_grad) + 1e-12)
            print(name, "relative error:", rel_err)


def train(W, lr):
    model = NN(W)
    batch = 32
    losses = []

    for epoch in range(50):
        idx = np.random.permutation(len(X_train))
        Xb = X_train[idx]
        Yb = Y_train[idx]

        for i in range(0, len(X_train), batch):
            X_batch = Xb[i:i+batch]
            Y_batch = Yb[i:i+batch]

            model.forward(X_batch)
            model.backward(X_batch, Y_batch)
            model.step(lr)

        y_hat = model.forward(X_train)
        loss = cross_entropy(y_hat, Y_train)
        losses.append(loss)
        print(f"Epoch {epoch+1}: loss={loss:.4f}")

    test_pred = np.argmax(model.forward(X_test), axis=1)
    test_true = np.argmax(Y_test, axis=1)
    acc = np.mean(test_pred == test_true)

    return losses, acc


def timing(model, X):
    f_times = []
    b_times = []

    for _ in range(100):
        t = time.time()
        model.forward(X)
        f_times.append(time.time() - t)

        t = time.time()
        model.backward(X, Y_train[:X.shape[0]])
        b_times.append(time.time() - t)

    return np.median(f_times), np.median(b_times)

loss_dict = {}
acc_dict = {}

for W in [16, 64]:
    for lr in [0.001, 0.01, 0.1]:
        losses, acc = train(W, lr)
        loss_dict[(W, lr)] = losses
        acc_dict[(W, lr)] = acc

plt.figure(figsize=(10,6))
for key, curve in loss_dict.items():
    W, lr = key
    plt.plot(curve, label=f"W={W}, lr={lr}")
plt.legend()
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Loss Curves")
plt.savefig("loss_curves.png")
