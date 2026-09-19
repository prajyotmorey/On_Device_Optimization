import torch
import time
import numpy as np
 
from models.teacher import TeacherCNN
from benchmarks.latency import benchmark_latency
from benchmarks.size import model_report
from benchmarks.accuracy import evaluate_accuracy
from benchmarks.sparsity import calculate_sparsity
from benchmarks.precision import get_model_precision

from torch.utils.data import DataLoader
from torchvision import datasets, transforms


def benchmark_model(model,dataloader,device,model_name):
    #accuracy_benchmatrking
    accuracy = evaluate_accuracy(model, dataloader, device)

    
    # Latency benchmarking
    results = []
    for i in range(10):
        input_tensor = torch.randn(1, 1, 28, 28)  # Batch size of 1, 1 channel, 28x28 image
        avg_latency_ms, fps = benchmark_latency(model, input_tensor, device)
        results.append((avg_latency_ms, fps))
        # print(f"Run {i+1}: Average Latency: {avg_latency_ms:.4f} ms | FPS: {fps:.2f}")
    
    ## latency Statistics
    latencies = np.array([r[0] for r in results])

    mean_latency = np.mean(latencies)
    std_latency = np.std(latencies)
    median_latency = np.median(latencies)

    p50 = np.percentile(latencies, 50)
    p95 = np.percentile(latencies, 95)
    p99 = np.percentile(latencies, 99)

    fps = 1000.0 / mean_latency  # Convert latency to FPS



    # Size benchmarking
    size_info = model_report(model)



    # Peak GPU memory usage benchmarking
    if device.type == 'cuda':
        torch.cuda.reset_peak_memory_stats()
        model.eval()
        x = torch.randn(1, 1, 28, 28, device="cuda")

        with torch.no_grad():
            _ = model(x)

        torch.cuda.synchronize()

        peak_memory = torch.cuda.max_memory_allocated()
    else:
        peak_memory = 0  # Not applicable for CPU


    sparsity = calculate_sparsity(model)
    precision = get_model_precision(model)

    return{
        "model": model_name,
        "parameters": size_info['parameters'],
        "trainable_parameters": size_info['trainable_parameters'],
        "size_mb": size_info['size_mb'],
        "sparsity": sparsity,
        "precision": precision,
        "accuracy": accuracy,
        "peak_memory_mb": peak_memory / (1024 ** 2),
        "mean_latency_ms": mean_latency,
        "std_latency_ms": std_latency,
        "median_latency_ms": median_latency,
        "p50_latency_ms": p50,
        "p95_latency_ms": p95,
        "p99_latency_ms": p99,
        "fps": fps
    }



