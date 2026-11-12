

class Value:
    def __init__(self, data, parents=(), op=''):
        self.data = data
        self.grad = 0.0
        self.parents = set(parents)
        self.op = op
        self._backward = lambda: None

    # Addition
    def __add__(self, other):
        if not isinstance(other, Value):
            other = Value(other)

        out = Value(self.data + other.data, (self, other), '+')

        def _backward():
            self.grad += out.grad
            other.grad += out.grad

        out._backward = _backward
        return out

    # Multiplication
    def __mul__(self, other):
        if not isinstance(other, Value):
            other = Value(other)

        out = Value(self.data * other.data, (self, other), '*')

        def _backward():
            self.grad += other.data * out.grad
            other.grad += self.data * out.grad

        out._backward = _backward
        return out

    # Power
    def __pow__(self, exponent):
        assert isinstance(exponent, (int, float)), \
            "Exponent must be an int or float"

        out = Value(
            self.data ** exponent,
            (self,),
            f'**{exponent}'
        )

        def _backward():
            self.grad += exponent * self.data ** (exponent - 1) * out.grad

        out._backward = _backward
        return out

    # ReLU
    def relu(self):
        out = Value(max(0, self.data), (self,), 'ReLU')

        def _backward():
            self.grad += (out.data > 0) * out.grad

        out._backward = _backward
        return out

    # Backward pass
    def backward(self):
        # Step 1: Topological sort
        topo = []
        visited = set()

        def build_topo(v):
            if v not in visited:
                visited.add(v)

                for parent in v.parents:
                    build_topo(parent)

                topo.append(v)

        build_topo(self)

        # Step 2: Set output gradient to 1
        self.grad = 1.0

        # Step 3: Visit nodes in reverse order
        for node in reversed(topo):
            # Step 4: Accumulate gradients
            node._backward()

    # Display
    def __repr__(self):
        return f"Value(data={self.data}, grad={self.grad})"
    
import torch

# Test inputs
a_val = 1.2
b_val = 0.7
c_val = 0.4

# -----------------------------------
# 1. YOUR IMPLEMENTATION
# -----------------------------------

a = Value(a_val)
b = Value(b_val)
c = Value(c_val)

output = ((a * b + c) ** 2).relu().sum()

output.backward()

my_grads = [a.grad, b.grad, c.grad]

# -----------------------------------
# 2. PYTORCH AUTOGRAD
# -----------------------------------

a_t = torch.tensor(a_val, dtype=torch.float64, requires_grad=True)
b_t = torch.tensor(b_val, dtype=torch.float64, requires_grad=True)
c_t = torch.tensor(c_val, dtype=torch.float64, requires_grad=True)

output_t = ((a_t * b_t + c_t) ** 2).relu().sum()

output_t.backward()

torch_grads = [
    a_t.grad.item(),
    b_t.grad.item(),
    c_t.grad.item()
]

# -----------------------------------
# 3. NUMERICAL GRADIENTS
# -----------------------------------

def function(a, b, c):
    return max(0.0, (a * b + c) ** 2)

def numerical_gradient(a, b, c, index, h=1e-6):
    values = [a, b, c]

    values[index] += h
    f_plus = function(*values)

    values[index] -= 2 * h
    f_minus = function(*values)

    return (f_plus - f_minus) / (2 * h)


numerical_grads = [
    numerical_gradient(a_val, b_val, c_val, 0),
    numerical_gradient(a_val, b_val, c_val, 1),
    numerical_gradient(a_val, b_val, c_val, 2)
]

# -----------------------------------
# 4. COMPARE RESULTS
# -----------------------------------

print("\nGradient Verification")
print("-" * 75)

print(f"{'Variable':<12}{'Your Grad':<18}"
      f"{'PyTorch Grad':<18}{'Numerical Grad':<18}")

for name, mine, pt, num in zip(
    ['a', 'b', 'c'],
    my_grads,
    torch_grads,
    numerical_grads
):
    print(f"{name:<12}{mine:<18.10f}"
          f"{pt:<18.10f}{num:<18.10f}")

# Maximum absolute differences
diff_torch = max(
    abs(x - y) for x, y in zip(my_grads, torch_grads)
)

diff_numerical = max(
    abs(x - y) for x, y in zip(my_grads, numerical_grads)
)

max_diff = max(diff_torch, diff_numerical)

print("\nMaximum difference vs PyTorch:", diff_torch)
print("Maximum difference vs Numerical:", diff_numerical)
print("Maximum absolute difference:", max_diff)

# -----------------------------------
# 5. ASSERTION
# -----------------------------------

assert max_diff < 1e-6, \
    "Gradient verification FAILED!"

print("\nSUCCESS: All gradient checks passed!")


    


# =================================
# TESTING THE IMPLEMENTATION
# =================================

a = Value(1.0)
b = Value(3.0)

# Addition
c = a + b
print("Addition:", c)

c.backward()
print("a gradient:", a.grad)
print("b gradient:", b.grad)
print("c gradient:", c.grad)

# Multiplication
a = Value(2.0)
b = Value(3.0)

d = a * b
print("\nMultiplication:", d)

# Power
e = a ** 2
print("Power:", e)

# ReLU
f = Value(-4.0).relu()
print("ReLU:", f)

# Backpropagation example
x = Value(2.0)
y = Value(3.0)

z = (x * y + x ** 2).relu()

z.backward()

print("\nBackpropagation:")
print("x:", x)
print("y:", y)
print("z:", z)
