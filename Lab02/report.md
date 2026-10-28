# Lab 02 — A Neural Network from Scratch
**Student:** Tori  
**Course:** Machine Learning Systems Engineering  
**Dataset:** Digits (8×8, 10 classes)

---

## 1. Goal
The goal of Lab 02 is to implement a two-hidden-layer neural network from scratch using NumPy, including forward propagation, cross-entropy loss, backpropagation, numerical gradient checking, and a full training loop. The trained models are evaluated for accuracy, learning dynamics, and forward/backward timing.

---

## 2. Method

### Dataset and Preprocessing
- Loaded the digits dataset using `load_digits(return_X_y=True)`.
- Scaled inputs by dividing pixel values by 16 to map them into [0, 1].
- One-hot encoded labels into 10-dimensional vectors.
- Performed a 70/30 stratified split with `random_state=42`.

### Network Architecture
- Input layer: 64 units (flattened 8×8 image).
- Hidden layer 1: width \(W \in \{16, 64\}\), ReLU activation.
- Hidden layer 2: width \(W\), ReLU activation.
- Output layer: 10 units, softmax activation.
- Weight shapes: (64, W), (W, W), (W, 10); bias shapes: (W,), (W,), (10,).
- Weights initialized from \( \mathcal{N}(0, 0.01) \); biases initialized to zero.

### Forward Pass and Loss
- Implemented forward propagation layer by layer:
  - \( z_1 = X W_1 + b_1 \), \( a_1 = \text{ReLU}(z_1) \)
  - \( z_2 = a_1 W_2 + b_2 \), \( a_2 = \text{ReLU}(z_2) \)
  - \( z_3 = a_2 W_3 + b_3 \), \( \hat{Y} = \text{softmax}(z_3) \)
- Verified that the output shape is (batch, 10) and each row sums to 1 within \(10^{-6}\).
- Used cross-entropy loss with one-hot labels; confirmed that the untrained network loss is approximately \(\ln(10) \approx 2.3026\).

### Backpropagation and Gradient Checking
- Derived gradients analytically using the chain rule for all weights and biases.
- Implemented backpropagation:
  - Used the identity \( \partial L / \partial z_3 = (\hat{Y} - Y) / N \) for softmax + cross-entropy.
  - Propagated gradients backward through ReLU activations and linear layers.
- Performed numerical gradient checking:
  - Selected 10 random weights.
  - Perturbed each by \(+10^{-5}\) and computed numerical slopes.
  - Compared analytic and numerical gradients; required relative error < \(10^{-6}\) for at least 9 of 10 weights.

### Training Loop
- Mini-batch size: 32.
- Epochs: 50.
- Learning rates: [0.001, 0.01, 0.1].
- Hidden widths: [16, 64].
- For each (W, learning rate) combination:
  - Shuffled training data each epoch.
  - Ran SGD updates on mini-batches.
  - Logged per-epoch loss and accuracy.
- Plotted loss curves to `results/loss_curves.png`.
- Recorded final test accuracy for each configuration.

### Timing
- Measured forward and backward pass times separately.
- Used the median of 100 runs on a fixed batch.
- Reported which pass dominates and related this to the number of matrix multiplications and gradient computations.

---

## 3. Results

### Accuracy and Learning Behaviour
Summarize here (fill in with your actual numbers):

- **W = 16, lr = 0.001:** final test accuracy = …
- **W = 16, lr = 0.01:** final test accuracy = …
- **W = 16, lr = 0.1:** final test accuracy = …
- **W = 64, lr = 0.001:** final test accuracy = …
- **W = 64, lr = 0.01:** final test accuracy = …
- **W = 64, lr = 0.1:** final test accuracy = …

Loss curves in `results/loss_curves.png` show how learning rate and width affect convergence speed and stability.

### Timing
- Forward pass median time (per batch): … ms
- Backward pass median time (per batch): … ms

Backward pass is expected to dominate due to additional matrix multiplications and gradient calculations.

---

## 4. Discussion

1. **Effect of hidden width:**  
   Increasing the hidden width from 16 to 64 generally improves representational capacity and can increase accuracy, but also increases computation and parameter count.

2. **Effect of learning rate:**  
   Very small learning rates converge slowly; very large learning rates may cause unstable or diverging loss. A mid-range learning rate (e.g. 0.01) typically balances speed and stability.

3. **Forward vs backward timing:**  
   The backward pass is slower than the forward pass because it requires computing gradients for every parameter, effectively performing additional matrix multiplications and elementwise operations.

---

## 5. Conclusions

1. A two-hidden-layer neural network with ReLU and softmax can be implemented from scratch using only NumPy, and trained to achieve reasonable accuracy on the digits dataset.
2. Gradient checking is essential for validating backpropagation; matching analytic and numerical gradients within tight tolerances builds confidence in the implementation.
3. Model width and learning rate strongly influence both accuracy and training dynamics, while backward pass timing highlights the computational cost of learning compared to inference.

---

# Lab 02 — A Neural Network from Scratch
**Student:** Tori  
**Course:** Machine Learning Systems Engineering  
**Dataset:** Digits (8×8, 10 classes)

---

## 1. Goal
The goal of Lab 02 is to implement a two-hidden-layer neural network from scratch using NumPy, including forward propagation, cross-entropy loss, backpropagation, numerical gradient checking, and a full training loop. The trained models are evaluated for accuracy, learning dynamics, and forward/backward timing.

---

## 2. Method

### Dataset and Preprocessing
- Loaded the digits dataset using `load_digits(return_X_y=True)`.
- Scaled inputs by dividing pixel values by 16 to map them into [0, 1].
- One-hot encoded labels into 10-dimensional vectors.
- Performed a 70/30 stratified split with `random_state=42`.

### Network Architecture
- Input layer: 64 units (flattened 8×8 image).
- Hidden layer 1: width \(W \in \{16, 64\}\), ReLU activation.
- Hidden layer 2: width \(W\), ReLU activation.
- Output layer: 10 units, softmax activation.
- Weight shapes: (64, W), (W, W), (W, 10); bias shapes: (W,), (W,), (10,).
- Weights initialized from \( \mathcal{N}(0, 0.01) \); biases initialized to zero.

### Forward Pass and Loss
- Implemented forward propagation layer by layer:
  - \( z_1 = X W_1 + b_1 \), \( a_1 = \text{ReLU}(z_1) \)
  - \( z_2 = a_1 W_2 + b_2 \), \( a_2 = \text{ReLU}(z_2) \)
  - \( z_3 = a_2 W_3 + b_3 \), \( \hat{Y} = \text{softmax}(z_3) \)
- Verified that the output shape is (batch, 10) and each row sums to 1 within \(10^{-6}\).
- Used cross-entropy loss with one-hot labels; confirmed that the untrained network loss is approximately \(\ln(10) \approx 2.3026\).

### Backpropagation and Gradient Checking
- Derived gradients analytically using the chain rule for all weights and biases.
- Implemented backpropagation:
  - Used the identity \( \partial L / \partial z_3 = (\hat{Y} - Y) / N \) for softmax + cross-entropy.
  - Propagated gradients backward through ReLU activations and linear layers.
- Performed numerical gradient checking:
  - Selected 10 random weights.
  - Perturbed each by \(+10^{-5}\) and computed numerical slopes.
  - Compared analytic and numerical gradients; required relative error < \(10^{-6}\) for at least 9 of 10 weights.

### Training Loop
- Mini-batch size: 32.
- Epochs: 50.
- Learning rates: [0.001, 0.01, 0.1].
- Hidden widths: [16, 64].
- For each (W, learning rate) combination:
  - Shuffled training data each epoch.
  - Ran SGD updates on mini-batches.
  - Logged per-epoch loss and accuracy.
- Plotted loss curves to `results/loss_curves.png`.
- Recorded final test accuracy for each configuration.

### Timing
- Measured forward and backward pass times separately.
- Used the median of 100 runs on a fixed batch.
- Reported which pass dominates and related this to the number of matrix multiplications and gradient computations.

---

## 3. Results

### Accuracy and Learning Behaviour
Summarize here (fill in with your actual numbers):

- **W = 16, lr = 0.001:** final test accuracy = …
- **W = 16, lr = 0.01:** final test accuracy = …
- **W = 16, lr = 0.1:** final test accuracy = …
- **W = 64, lr = 0.001:** final test accuracy = …
- **W = 64, lr = 0.01:** final test accuracy = …
- **W = 64, lr = 0.1:** final test accuracy = …

Loss curves in `results/loss_curves.png` show how learning rate and width affect convergence speed and stability.

### Timing
- Forward pass median time (per batch): … ms
- Backward pass median time (per batch): … ms

Backward pass is expected to dominate due to additional matrix multiplications and gradient calculations.

---

## 4. Discussion

1. **Effect of hidden width:**  
   Increasing the hidden width from 16 to 64 generally improves representational capacity and can increase accuracy, but also increases computation and parameter count.

2. **Effect of learning rate:**  
   Very small learning rates converge slowly; very large learning rates may cause unstable or diverging loss. A mid-range learning rate (e.g. 0.01) typically balances speed and stability.

3. **Forward vs backward timing:**  
   The backward pass is slower than the forward pass because it requires computing gradients for every parameter, effectively performing additional matrix multiplications and elementwise operations.

---

## 5. Conclusions

1. A two-hidden-layer neural network with ReLU and softmax can be implemented from scratch using only NumPy, and trained to achieve reasonable accuracy on the digits dataset.
2. Gradient checking is essential for validating backpropagation; matching analytic and numerical gradients within tight tolerances builds confidence in the implementation.
3. Model width and learning rate strongly influence both accuracy and training dynamics, while backward pass timing highlights the computational cost of learning compared to inference.

---
