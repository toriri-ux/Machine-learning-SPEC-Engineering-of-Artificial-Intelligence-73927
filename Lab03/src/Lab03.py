
import time
import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn

import torchvision
import torchvision.transforms as transforms

from torch.utils.data import DataLoader, random_split

from torchinfo import summary
from thop import profile

# DEVICE

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("Using Device:", device)

# DATASET

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.2860,), (0.3530,))
])

train_dataset_full = torchvision.datasets.FashionMNIST(
    root="./data",
    train=True,
    download=True,
    transform=transform
)

test_dataset = torchvision.datasets.FashionMNIST(
    root="./data",
    train=False,
    download=True,
    transform=transform
)

train_dataset, val_dataset = random_split(
    train_dataset_full,
    [55000, 5000]
)

train_loader = DataLoader(
    train_dataset,
    batch_size=128,
    shuffle=True
)

val_loader = DataLoader(
    val_dataset,
    batch_size=128,
    shuffle=False
)

test_loader = DataLoader(
    test_dataset,
    batch_size=128,
    shuffle=False
)

print("Train:", len(train_dataset))
print("Validation:", len(val_dataset))
print("Test:", len(test_dataset))

# MLP MODEL

class MLP(nn.Module):
    def __init__(self):
        super().__init__()

        self.network = nn.Sequential(
            nn.Flatten(),

            nn.Linear(784, 256),
            nn.ReLU(),

            nn.Linear(256, 128),
            nn.ReLU(),

            nn.Linear(128, 10)
        )

    def forward(self, x):
        return self.network(x)

# CNN MODEL

class CNN(nn.Module):
    def __init__(self):
        super().__init__()

        self.features = nn.Sequential(
            nn.Conv2d(
                1,
                16,
                kernel_size=3,
                padding=1
            ),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(
                16,
                32,
                kernel_size=3,
                padding=1
            ),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )

        self.classifier = nn.Linear(
            32 * 7 * 7,
            10
        )

    def forward(self, x):

        x = self.features(x)

        x = torch.flatten(x, start_dim=1)

        x = self.classifier(x)

        return x

# ATTENTION MODEL

class AttentionModel(nn.Module):
    def __init__(self):
        super().__init__()

        self.patch_size = 7
        self.patch_dim = 49

        self.patch_embedding = nn.Linear(
            self.patch_dim,
            64
        )

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=64,
            nhead=4,
            dim_feedforward=128,
            batch_first=True
        )

        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=2
        )

        self.fc = nn.Linear(64, 10)

    def forward(self, x):

        batch_size = x.size(0)

        patches = (
            x.unfold(2, 7, 7)
             .unfold(3, 7, 7)
        )

        patches = patches.contiguous().view(
            batch_size,
            16,
            49
        )

        x = self.patch_embedding(patches)

        x = self.transformer(x)

        x = x.mean(dim=1)

        x = self.fc(x)

        return x

# TRAINING FUNCTION

def train_model(
        model,
        train_loader,
        val_loader,
        epochs=5,
        lr=0.001):

    model.to(device)

    criterion = nn.CrossEntropyLoss()

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=lr
    )

    train_losses = []
    val_accuracies = []

    for epoch in range(epochs):

        model.train()

        running_loss = 0

        for images, labels in train_loader:

            images = images.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()

            outputs = model(images)

            loss = criterion(
                outputs,
                labels
            )

            loss.backward()

            optimizer.step()

            running_loss += loss.item()

        avg_loss = (
            running_loss /
            len(train_loader)
        )

        train_losses.append(avg_loss)

        val_acc = evaluate(
            model,
            val_loader
        )

        val_accuracies.append(val_acc)

        print(
            f"Epoch {epoch+1}/{epochs} | "
            f"Loss: {avg_loss:.4f} | "
            f"Val Acc: {val_acc:.2f}%"
        )

    return train_losses, val_accuracies

# EVALUATION

def evaluate(model, loader):

    model.eval()

    correct = 0
    total = 0

    with torch.no_grad():

        for images, labels in loader:

            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)

            predictions = torch.argmax(
                outputs,
                dim=1
            )

            correct += (
                predictions == labels
            ).sum().item()

            total += labels.size(0)

    return 100 * correct / total

# LATENCY FUNCTION

def measure_latency(
        model,
        input_tensor,
        runs):

    model.eval()

    times = []

    with torch.no_grad():

        for _ in range(runs):

            start = time.perf_counter()

            model(input_tensor)

            end = time.perf_counter()

            times.append(
                (end - start) * 1000
            )

    return np.median(times)

# CREATE MODELS

mlp = MLP()
cnn = CNN()
attn = AttentionModel()

# TRAIN MODELS

print("\nTraining MLP")
train_model(
    mlp,
    train_loader,
    val_loader,
    epochs=5
)

print("\nTraining CNN")
train_model(
    cnn,
    train_loader,
    val_loader,
    epochs=5
)

print("\nTraining Attention")
train_model(
    attn,
    train_loader,
    val_loader,
    epochs=5
)

# TEST ACCURACY

mlp_acc = evaluate(
    mlp,
    test_loader
)

cnn_acc = evaluate(
    cnn,
    test_loader
)

attn_acc = evaluate(
    attn,
    test_loader
)

print("\nTEST RESULTS")

print(f"MLP Accuracy: {mlp_acc:.2f}%")
print(f"CNN Accuracy: {cnn_acc:.2f}%")
print(f"Attention Accuracy: {attn_acc:.2f}%")


# PARAMETER COUNTS

print("\nMODEL SUMMARIES")

summary(
    mlp,
    input_size=(1,1,28,28)
)

summary(
    cnn,
    input_size=(1,1,28,28)
)

summary(
    attn,
    input_size=(1,1,28,28)
)


# FLOP PROFILING


dummy_input = torch.randn(
    1,
    1,
    28,
    28
)

mlp_flops, mlp_params = profile(
    mlp,
    inputs=(dummy_input,),
    verbose=False
)

cnn_flops, cnn_params = profile(
    cnn,
    inputs=(dummy_input,),
    verbose=False
)

attn_flops, attn_params = profile(
    attn,
    inputs=(dummy_input,),
    verbose=False
)

print("\nFLOPS")

print("MLP")
print("FLOPs:", mlp_flops)
print("Params:", mlp_params)

print("\nCNN")
print("FLOPs:", cnn_flops)
print("Params:", cnn_params)

print("\nAttention")
print("FLOPs:", attn_flops)
print("Params:", attn_params)

# LATENCY

x1 = torch.randn(
    1,
    1,
    28,
    28
).to(device)

x128 = torch.randn(
    128,
    1,
    28,
    28
).to(device)

mlp.to(device)
cnn.to(device)
attn.to(device)

print("\nLATENCY")

mlp_latency = measure_latency(
    mlp,
    x1,
    100
)

cnn_latency = measure_latency(
    cnn,
    x1,
    100
)

attn_latency = measure_latency(
    attn,
    x1,
    100
)

print("MLP:", mlp_latency, "ms")
print("CNN:", cnn_latency, "ms")
print("Attention:", attn_latency, "ms")


# GRAPHS

accuracies = [
    mlp_acc,
    cnn_acc,
    attn_acc
]

params = [
    mlp_params,
    cnn_params,
    attn_params
]

latencies = [
    mlp_latency,
    cnn_latency,
    attn_latency
]

models = [
    "MLP",
    "CNN",
    "Attention"
]

# Accuracy vs Parameters

plt.figure(figsize=(8,6))

plt.scatter(
    params,
    accuracies,
    s=120
)

for i in range(len(models)):
    plt.annotate(
        models[i],
        (params[i], accuracies[i])
    )

plt.xlabel("Parameters")
plt.ylabel("Accuracy (%)")
plt.title("Accuracy vs Parameters")
plt.grid(True)
plt.savefig("Accuracy_VS_Parameters.png")
plt.show()



# Accuracy vs Latency
plt.figure(figsize=(8,6))

plt.scatter(
    latencies,
    accuracies,
    s=120
)

for i in range(len(models)):
    plt.annotate(
        models[i],
        (latencies[i], accuracies[i])
    )

plt.xlabel("Latency (ms)")
plt.ylabel("Accuracy (%)")
plt.title("Accuracy vs Latency")
plt.grid(True)
plt.savefig("Accuracy_VS_Latency.png")
plt.show()

# MANUAL FLOP CALCULATIONS

mlp_manual = 2 * 784 * 256

cnn_manual = (
    2 * 3 * 3 * 1 * 16 * 28 * 28
)

attention_manual = (
    2 * 16 * 64 * 3 * 64
)

print("\n MANUAL FLOPS ")

print(
    "MLP Linear(784->256):",
    mlp_manual
)

print(
    "CNN Conv2D:",
    cnn_manual
)

print(
    "Attention QKV:",
    attention_manual
)

# SAVE RESULTS TO FILE

with open("Results_Report.txt", "w") as f:

    f.write("FASHION-MNIST MODEL COMPARISON REPORT\n")
    f.write("="*50 + "\n\n")

    # Accuracy Results
    f.write("TEST ACCURACY\n")
    f.write("-"*30 + "\n")
    f.write(f"MLP Accuracy      : {mlp_acc:.2f}%\n")
    f.write(f"CNN Accuracy      : {cnn_acc:.2f}%\n")
    f.write(f"Attention Accuracy: {attn_acc:.2f}%\n\n")

    # Parameters
    f.write("PARAMETER COUNTS\n")
    f.write("-"*30 + "\n")
    f.write(f"MLP Parameters      : {int(mlp_params):,}\n")
    f.write(f"CNN Parameters      : {int(cnn_params):,}\n")
    f.write(f"Attention Parameters: {int(attn_params):,}\n\n")

    # FLOPs
    f.write("THOP FLOP RESULTS\n")
    f.write("-"*30 + "\n")
    f.write(f"MLP FLOPs      : {mlp_flops:,.0f}\n")
    f.write(f"CNN FLOPs      : {cnn_flops:,.0f}\n")
    f.write(f"Attention FLOPs: {attn_flops:,.0f}\n\n")

    # Latency
    f.write("LATENCY RESULTS\n")
    f.write("-"*30 + "\n")
    f.write(f"MLP Latency      : {mlp_latency:.4f} ms\n")
    f.write(f"CNN Latency      : {cnn_latency:.4f} ms\n")
    f.write(f"Attention Latency: {attn_latency:.4f} ms\n\n")

    # Manual FLOPs
    f.write("MANUAL FLOP CALCULATIONS\n")
    f.write("-"*30 + "\n")
    f.write(f"MLP Linear Layer      : {mlp_manual:,}\n")
    f.write(f"CNN Conv Layer        : {cnn_manual:,}\n")
    f.write(f"Attention QKV Layer   : {attention_manual:,}\n\n")

    # Best Model
    accuracies = {
        "MLP": mlp_acc,
        "CNN": cnn_acc,
        "Attention": attn_acc
    }

    best_model = max(accuracies, key=accuracies.get)

    f.write("CONCLUSION\n")
    f.write("-"*30 + "\n")
    f.write(
        f"Best Accuracy Model: {best_model} "
        f"({accuracies[best_model]:.2f}%)\n"
    )

