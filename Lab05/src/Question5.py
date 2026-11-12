import time
import random
import statistics
import numpy as np
import torch
import torch.nn as nn
from sklearn.datasets import make_moons

class Value:
    def __init__(self, data, parents=(), op=''):
        self.data = float(data)
        self.grad = 0.0
        self.parents = set(parents)
        self.op = op
        self._backward = lambda: None

    def __add__(self, other):
        if not isinstance(other, Value):
            other = Value(other)

        out = Value(self.data + other.data, (self, other), '+')

        def _backward():
            self.grad += out.grad
            other.grad += out.grad

        out._backward = _backward
        return out

    __radd__ = __add__

    def __neg__(self):
        return self * -1

    def __sub__(self, other):
        return self + (-other if isinstance(other, Value) else -float(other))

    def __rsub__(self, other):
        return (-self) + other

    def __mul__(self, other):
        if not isinstance(other, Value):
            other = Value(other)

        out = Value(self.data * other.data, (self, other), '*')

        def _backward():
            self.grad += other.data * out.grad
            other.grad += self.data * out.grad

        out._backward = _backward
        return out

    __rmul__ = __mul__

    def __truediv__(self, other):
        if not isinstance(other, Value):
            other = Value(other)
        return self * (other ** -1)

    def __pow__(self, exponent):
        out = Value(
            self.data ** exponent,
            (self,),
            f'**{exponent}'
        )

        def _backward():
            self.grad += (
                exponent * self.data ** (exponent - 1)
            ) * out.grad

        out._backward = _backward
        return out

    def exp(self):
        out = Value(np.exp(self.data), (self,), 'exp')

        def _backward():
            self.grad += out.data * out.grad

        out._backward = _backward
        return out

    def log(self):
        out = Value(np.log(self.data), (self,), 'log')

        def _backward():
            self.grad += (1.0 / self.data) * out.grad

        out._backward = _backward
        return out

    def relu(self):
        out = Value(max(0.0, self.data), (self,), 'ReLU')

        def _backward():
            self.grad += (self.data > 0) * out.grad

        out._backward = _backward
        return out

    def sigmoid(self):
        return 1.0 / (1.0 + (-self).exp())

    def backward(self):
        topo = []
        visited = set()

        def build_topo(v):
            if v not in visited:
                visited.add(v)
                for parent in v.parents:
                    build_topo(parent)
                topo.append(v)

        build_topo(self)

        self.grad = 1.0

        for node in reversed(topo):
            node._backward()

    def __repr__(self):
        return f"Value(data={self.data}, grad={self.grad})"



X, y = make_moons(
    n_samples=200,
    noise=0.2,
    random_state=42
)

X = X.astype(np.float64)
y = y.astype(np.float64)

X_t = torch.tensor(X, dtype=torch.float64)
y_t = torch.tensor(y.reshape(-1, 1), dtype=torch.float64)



def initialize_parameters(seed=42):
    rng = random.Random(seed)

    W1 = [
        [rng.uniform(-0.5, 0.5) for _ in range(2)]
        for _ in range(8)
    ]

    b1 = [rng.uniform(-0.1, 0.1) for _ in range(8)]

    W2 = [
        [rng.uniform(-0.5, 0.5) for _ in range(8)]
    ]

    b2 = [rng.uniform(-0.1, 0.1)]

    return W1, b1, W2, b2



def train_custom(epochs=30, lr=0.1):

    W1_data, b1_data, W2_data, b2_data = \
        initialize_parameters()

    W1 = [[Value(v) for v in row] for row in W1_data]
    b1 = [Value(v) for v in b1_data]

    W2 = [[Value(v) for v in row] for row in W2_data]
    b2 = [Value(v) for v in b2_data]

    params = (
        [p for row in W1 for p in row]
        + b1
        + [p for row in W2 for p in row]
        + b2
    )

    final_loss = None

    start = time.perf_counter()

    for epoch in range(epochs):

        # Reset parameter gradients
        for p in params:
            p.grad = 0.0

        total_loss = Value(0.0)

        # Forward pass
        for xi, target in zip(X, y):

            inputs = [Value(v) for v in xi]

            # Hidden layer: 2 -> 8
            hidden = []

            for j in range(8):
                z = b1[j]

                for k in range(2):
                    z = z + W1[j][k] * inputs[k]

                hidden.append(z.sigmoid())

            # Output layer: 8 -> 1
            z = b2[0]

            for j in range(8):
                z = z + W2[0][j] * hidden[j]

            prediction = z.sigmoid()

            # Binary cross-entropy
            loss = -(
                target * prediction.log()
                + (1 - target) * (1 - prediction).log()
            )

            total_loss = total_loss + loss

        # Mean loss
        total_loss = total_loss / len(X)

        # Backward pass
        total_loss.backward()

        # SGD update
        for p in params:
            p.data -= lr * p.grad

        final_loss = total_loss.data

    elapsed = time.perf_counter() - start

    return elapsed, final_loss


def train_pytorch(epochs=30, lr=0.1):

    W1, b1, W2, b2 = initialize_parameters()

    model = nn.Sequential(
        nn.Linear(2, 8),
        nn.Sigmoid(),
        nn.Linear(8, 1),
        nn.Sigmoid()
    ).double()

    with torch.no_grad():
        model[0].weight.copy_(
            torch.tensor(W1, dtype=torch.float64)
        )
        model[0].bias.copy_(
            torch.tensor(b1, dtype=torch.float64)
        )
        model[2].weight.copy_(
            torch.tensor(W2, dtype=torch.float64)
        )
        model[2].bias.copy_(
            torch.tensor(b2, dtype=torch.float64)
        )

    criterion = nn.BCELoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=lr)

    final_loss = None

    start = time.perf_counter()

    for epoch in range(epochs):

        optimizer.zero_grad()

        predictions = model(X_t)
        loss = criterion(predictions, y_t)

        loss.backward()
        optimizer.step()

        final_loss = loss.item()

    elapsed = time.perf_counter() - start

    return elapsed, final_loss

torch.set_num_threads(1)

RUNS = 5
EPOCHS = 30

custom_times = []
torch_times = []

custom_losses = []
torch_losses = []

for run in range(RUNS):

    custom_time, custom_loss = train_custom(EPOCHS)
    torch_time, torch_loss = train_pytorch(EPOCHS)

    custom_times.append(custom_time)
    torch_times.append(torch_time)

    custom_losses.append(custom_loss)
    torch_losses.append(torch_loss)

    print(f"\nRun {run + 1}")
    print(f"Custom Autograd: {custom_time:.6f} seconds")
    print(f"PyTorch:         {torch_time:.6f} seconds")




custom_median = statistics.median(custom_times)
torch_median = statistics.median(torch_times)

print("\n" + "=" * 50)
print("BENCHMARK RESULTS (MEDIAN OF 5 RUNS)")
print("=" * 50)

print(f"Custom Autograd: {custom_median:.6f} seconds")
print(f"PyTorch:         {torch_median:.6f} seconds")

print(f"\nCustom final loss: {statistics.median(custom_losses):.6f}")
print(f"PyTorch final loss: {statistics.median(torch_losses):.6f}")

print(f"\nPyTorch speedup: {custom_median / torch_median:.2f}x")

print("\nBenchmark completed!")
