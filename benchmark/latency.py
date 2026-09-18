import time
import torch

@torch.no_grad() #temporarily disable gradient calculation for inference

def benchmark_latency(model, input_tensor, device, warmup=50, num_iterations=200):
    model.eval() #inference mode
    input_tensor = input_tensor.to(device)

    # Warm-up iterations
    for _ in range(warmup):
        _ = model(input_tensor)


    if device.type == 'cuda':
        torch.cuda.synchronize()  # Forces CPU to wait untill all kernal calls on the GPU have completed.

    start = time.pref_counter()

    for _ in range(num_iterations):
        _ = model(input_tensor)

    if device.type == 'cuda':
        torch.cuda.synchronize()  # Forces CPU to wait.

    end = time.perf_counter()

    avg_latency_ms = ((end - start) / num_iterations) * 1000  # Convert seconds to milliseconds

    fps = 1000 / (end - start)  # Frames per second

    return avg_latency_ms, fps
