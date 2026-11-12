import torch
import torch.nn as nn
import torch.fx as fx
import time
import statistics

torch.manual_seed(42)
torch.set_num_threads(1)

DEVICE = "cpu"
DTYPE = torch.float32

WARMUP = 20
RUNS = 5
ITERATIONS = 100

print("PyTorch version:", torch.__version__)
print("Device:", DEVICE)


class SmallModel(nn.Module):
    def __init__(self):
        super().__init__()

        self.fc1 = nn.Linear(2, 8)
        self.fc2 = nn.Linear(8, 1)

    def forward(self, x):
        x = torch.sigmoid(self.fc1(x))
        x = torch.sigmoid(self.fc2(x))
        return x


model = SmallModel().to(DEVICE).eval()

# Sample input
x = torch.randn(1024, 2, device=DEVICE, dtype=DTYPE)


print("\n========== FX COMPUTATION GRAPH ==========")

fx_model = fx.symbolic_trace(model)

print(fx_model.graph)
print("\nGraph code:")
print(fx_model.code)

print("\n========== TENSOR INSPECTION ==========")

def inspect_tensor(name, tensor):
    print(f"\n{name}")
    print("Shape:", tensor.shape)
    print("Dtype:", tensor.dtype)
    print("Stride:", tensor.stride())
    print("Contiguous:", tensor.is_contiguous())

inspect_tensor("Input tensor", x)

with torch.no_grad():
    output = model(x)

inspect_tensor("Output tensor", output)


def benchmark(fn, input_tensor, warmup=WARMUP,
              runs=RUNS, iterations=ITERATIONS):

    # Warm-up: compilation and initialization
    with torch.inference_mode():
        for _ in range(warmup):
            fn(input_tensor)

    times = []

    # Measure five independent runs
    with torch.inference_mode():
        for _ in range(runs):

            start = time.perf_counter()

            for _ in range(iterations):
                fn(input_tensor)

            elapsed = time.perf_counter() - start

            times.append(elapsed / iterations)

    return statistics.median(times) * 1000



print("\n========== EAGER EXECUTION ==========")

eager_fn = model

eager_time = benchmark(eager_fn, x)

print(f"Eager median: {eager_time:.6f} ms")

print("\n========== TORCH.COMPILE ==========")

compiled_fn = None
compiled_time = None
compiled_supported = False

try:
    compiled_fn = torch.compile(model)

    # First call triggers compilation
    with torch.inference_mode():
        compiled_output = compiled_fn(x)

    # Check correctness
    torch.testing.assert_close(
        compiled_output,
        output,
        rtol=1e-4,
        atol=1e-5
    )

    compiled_time = benchmark(compiled_fn, x)
    compiled_supported = True

    print(f"Compiled median: {compiled_time:.6f} ms")

except Exception as error:
    print("torch.compile unavailable or failed:")
    print(type(error).__name__, str(error)[:500])


traced_fn = None
traced_time = None

if not compiled_supported:

    print("\n========== TORCH.JIT.TRACE ==========")

    with torch.inference_mode():
        traced_fn = torch.jit.trace(model, x)

        traced_output = traced_fn(x)

    torch.testing.assert_close(
        traced_output,
        output,
        rtol=1e-4,
        atol=1e-5
    )

    print("\nTraced computation graph:")
    print(traced_fn.graph)

    traced_time = benchmark(traced_fn, x)

    print(f"Traced median: {traced_time:.6f} ms")



    
    print("\n========== EXECUTION COMPARISON ==========")
    
    print(f"Eager: {eager_time:.6f} ms")
    
    if compiled_supported:
        print(f"Compiled: {compiled_time:.6f} ms")
    
        print(
            "Compile speedup:",
            f"{eager_time / compiled_time:.3f}x"
        )
    else:
        print(f"JIT Trace: {traced_time:.6f} ms")
    
        print(
            "Trace speedup:",
            f"{eager_time / traced_time:.3f}x"
        )
    print("\n========== MEMORY LAYOUT EXPERIMENT ==========")
    
  
# Create base tensor
base = torch.randn(256, 4096, dtype=DTYPE)

# Non-contiguous view: shape (4096, 256)
non_contiguous_tensor = base.t()

# Contiguous copy with exactly the same values and shape
contiguous_tensor = non_contiguous_tensor.contiguous()

# Matrix multiplication weight
weight = torch.randn(256, 128, dtype=DTYPE)

print("Same shape:",
      contiguous_tensor.shape == non_contiguous_tensor.shape)

print("Same values:",
      torch.equal(contiguous_tensor, non_contiguous_tensor))

print("Contiguous:",
      contiguous_tensor.is_contiguous())

print("Non-contiguous:",
      non_contiguous_tensor.is_contiguous())

print("\nSame shape:",
          contiguous_tensor.shape == non_contiguous_tensor.shape)
    
    
    # Operation: matrix multiplication
def matrix_operation(tensor):
        return torch.matmul(tensor, weight)
    
    
    # Warm-up and benchmark
contiguous_time = benchmark(
        matrix_operation,
        contiguous_tensor
    )
    
non_contiguous_time = benchmark(
        matrix_operation,
        non_contiguous_tensor
    )
    
print("\n========== MEMORY PERFORMANCE ==========")
    
print(f"Contiguous: {contiguous_time:.6f} ms")
print(f"Non-contiguous: {non_contiguous_time:.6f} ms")

print(
    "Non-contiguous / contiguous ratio:",
    f"{non_contiguous_time / contiguous_time:.3f}x"
)
contiguous_copy = non_contiguous_tensor.contiguous()
    
inspect_tensor("Converted contiguous tensor", contiguous_copy)
    
copy_time = benchmark(
        matrix_operation,
        contiguous_copy
    )
    
print(f"\nAfter .contiguous(): {copy_time:.6f} ms")
    
print("\n========== FINAL SUMMARY ==========")

print("Eager execution:", round(eager_time, 6), "ms")

if compiled_supported:
    print("Compiled execution:", round(compiled_time, 6), "ms")
else:
    print("JIT traced execution:", round(traced_time, 6), "ms")
    
    print("Contiguous operation:", round(contiguous_time, 6), "ms")
    print("Non-contiguous operation:", round(non_contiguous_time, 6), "ms")
    print("Contiguous copy operation:", round(copy_time, 6), "ms")
    
    print("\nAll experiments completed!")