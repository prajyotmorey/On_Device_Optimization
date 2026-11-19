# On-Device Model Compression Pipeline

A practical PyTorch project exploring how to reduce neural-network model size and computational footprint while preserving predictive accuracy. The project implements and evaluates **magnitude pruning, masked fine-tuning, symmetric quantization, activation calibration, and knowledge distillation** on Fashion-MNIST.

> **Project motivation:** Consider a model with a 150 MB footprint and 200 ms inference latency per frame that must run at 30 FPS on a phone. This project investigates the compression techniques and measurement workflow needed to approach such deployment constraints. The experiments in this repository use Fashion-MNIST and an NVIDIA Tesla T4; they are not claims of achieved mobile deployment.

---

## 1. Project goals

The project is designed around the following engineering questions:

- How much can a CNN be compressed before accuracy degrades significantly?
- How does global magnitude pruning behave at different sparsity levels?
- Does gradual pruning with a cubic schedule and masked fine-tuning preserve accuracy better than one-shot pruning?
- How do per-tensor and per-channel symmetric quantization compare?
- How can activation ranges be calibrated and quantization error characterized?
- Can a smaller student model learn from a larger teacher through knowledge distillation?
- What does each compression lever contribute to model size, accuracy, latency, and memory?
- How should these methods be benchmarked consistently rather than judged by model size or sparsity alone?

### Deployment target and measurement caveat

| Item | Motivating target |
|---|---:|
| Model size | 150 MB |
| Inference latency | 200 ms/frame |
| Required throughput | 30 FPS |
| Frame-time budget at 30 FPS | 33.33 ms/frame |

The 200 ms/frame starting point is approximately six times slower than the 33.33 ms/frame budget. These figures define the motivating scenario. **The experiments reported below were measured on an NVIDIA Tesla T4, not on a phone.** No mobile latency, power, thermal, or end-to-end deployment result is claimed.

---

## 2. Experimental environment and dataset

| Component | Details |
|---|---|
| Framework | PyTorch 2.14.0 |
| Dataset | Fashion-MNIST |
| Input tensor | 1 × 28 × 28 |
| Number of classes | 10 |
| GPU used for reported baseline measurements | NVIDIA Tesla T4 |
| Teacher model | Custom CNN |
| Student model | Narrower custom CNN |
| Precision explored | FP32 baseline; custom INT8-style quantization analysis |
| Main evaluation | Test accuracy, parameters, serialized/model footprint estimate, sparsity, latency distribution, peak GPU memory, quantization error |

The teacher was trained for 10 epochs. The reported teacher test accuracy was approximately **85.4%** (85.44% after training; 85.38% in the baseline benchmark).

---

## 3. Model architectures

### Teacher CNN

The teacher uses three convolutional stages followed by global adaptive average pooling and a linear classifier.

| Layer | Configuration |
|---|---|
| Conv 1 | 1 → 32 channels, 3 × 3, padding 1 |
| Activation | ReLU |
| Downsampling | 2 × 2 max pooling |
| Conv 2 | 32 → 64 channels, 3 × 3, padding 1 |
| Activation | ReLU |
| Downsampling | 2 × 2 max pooling |
| Conv 3 | 64 → 128 channels, 3 × 3, padding 1 |
| Activation | ReLU |
| Pooling | Adaptive average pooling to 1 × 1 |
| Classifier | Linear, 128 → 10 |

### Student CNN

The student follows the same general design but uses narrower channel widths.

| Layer | Configuration |
|---|---|
| Conv 1 | 1 → 16 channels, 3 × 3, padding 1 |
| Activation | ReLU |
| Downsampling | 2 × 2 max pooling |
| Conv 2 | 16 → 32 channels, 3 × 3, padding 1 |
| Activation | ReLU |
| Downsampling | 2 × 2 max pooling |
| Conv 3 | 32 → 64 channels, 3 × 3, padding 1 |
| Activation | ReLU |
| Pooling | Adaptive average pooling to 1 × 1 |
| Classifier | Linear, 64 → 10 |

---

## 4. Baseline characterization

A unified benchmark workflow was established to characterize models consistently. The teacher baseline was evaluated with batch size 1 on the NVIDIA Tesla T4.

| Metric | Teacher FP32 baseline |
|---|---:|
| Parameters | 93,962 |
| Trainable parameters | 93,962 |
| Model size | 0.35844 MB |
| Test accuracy | 85.38% |
| Mean latency | 0.33189 ms |
| Median / P50 latency | 0.32016 ms |
| Standard deviation | 0.02563 ms |
| P95 latency | 0.37574 ms |
| P99 latency | 0.39624 ms |
| Throughput derived from mean latency | ~3,013 FPS |
| Peak allocated GPU memory | 10.666 MB |

**Interpretation:** The teacher is a compact Fashion-MNIST CNN, not a 150 MB model. The benchmark is a T4 reference measurement and should not be interpreted as phone performance.

### Benchmarking implementation

The project includes reusable benchmark modules:

| Module | Responsibility |
|---|---|
| `benchmarks/accuracy.py` | Evaluate test accuracy |
| `benchmarks/size.py` | Count parameters and estimate model size |
| `benchmarks/latency.py` | Measure latency distribution and FPS |
| `benchmarks/sparsity.py` | Calculate global and layer-wise sparsity |
| `benchmarks/precision.py` | Inspect parameter precision |
| `benchmarks/memory.py` | Measure peak GPU memory |
| `benchmarks/benchmark.py` | Combine benchmark metrics |
| `benchmarks/run_baseline.py` | Run baseline evaluation |

Latency benchmarking uses warm-up iterations, repeated timed iterations, batch size 1, and explicit CUDA synchronization. The recorded metrics include mean, median/P50, standard deviation, P95, P99, and FPS. Comparisons should use the same hardware, input shape, batch size, warm-up, and iteration settings.

---

## 5. Experiment 1 — Global magnitude pruning

Global magnitude pruning was implemented over the weights of all `Conv2d` and `Linear` layers:

1. Collect absolute values of eligible weights across the model.
2. Compute a global magnitude threshold for the target sparsity.
3. Set weights at or below the threshold to zero.
4. Evaluate the pruned model.

### One-shot pruning results

Each target was applied starting from the original teacher checkpoint.

| Target sparsity | Observed sparsity | Test accuracy | Dense FP32 size |
|---:|---:|---:|---:|
| 10% | 9.98% | 85.38% | 0.3584 MB |
| 20% | 19.95% | 85.38% | 0.3584 MB |
| 30% | 29.93% | 85.40% | 0.3584 MB |
| 40% | 39.90% | 85.60% | 0.3584 MB |
| 50% | 49.88% | 85.37% | 0.3584 MB |
| 60% | 59.85% | 84.33% | 0.3584 MB |
| 70% | 69.83% | 78.66% | 0.3584 MB |
| 80% | 79.80% | 34.23% | 0.3584 MB |
| 90% | 89.78% | 10.15% | 0.3584 MB |

### Layer-wise sparsity at approximately 50% global sparsity

| Layer | Observed sparsity |
|---|---:|
| First convolution | 15.28% |
| Second convolution | 39.88% |
| Third convolution | 52.98% |
| Classifier | 32.03% |

At approximately 40% global sparsity, layer-wise sparsities were 11.81%, 31.94%, 42.41%, and 23.67%, respectively.

### Analysis

- Accuracy remained broadly stable through 50% one-shot sparsity in this experiment.
- Accuracy degraded substantially at 70% and above, with severe collapse at 80–90%.
- Global pruning did not distribute zeros evenly across layers. Layers with different weight distributions received different sparsity levels.
- **Unstructured sparsity did not reduce the dense FP32 model footprint:** zero-valued weights were still stored in dense tensors.
- A dense implementation does not automatically become faster because some weights are zero. Sparse storage formats and sparse-kernel/runtime support are needed to realize storage or latency benefits.
- Measured latency was not consistently improved by increasing sparsity.

---

## 6. Experiment 2 — Cubic sparsity scheduling and masked fine-tuning

A gradual pruning approach was implemented using a cubic sparsity schedule, followed by continued training while enforcing masks after optimizer updates.

The target was 70% sparsity, reached at epoch index 6. Pruning was then held at 70% while fine-tuning continued.

| Epoch | Target sparsity | Threshold | Test accuracy |
|---:|---:|---:|---:|
| 0 | 0.00% | 0.0000 | 85.65% |
| 1 | 29.49% | 0.0063 | 86.84% |
| 2 | 49.26% | 0.0579 | 87.16% |
| 3 | 61.25% | 0.1285 | 86.15% |
| 4 | 67.41% | 0.2431 | 82.64% |
| 5 | 69.68% | 0.4359 | 70.02% |
| 6 | 70.00% | 0.6827 | 59.27% |
| 7 | 70.00% | — | 63.20% |
| 8 | 70.00% | — | 64.98% |
| 9 | 70.00% | — | 66.45% |

The final three epochs were fixed-mask fine-tuning epochs. Accuracy recovered from 59.27% at the target-sparsity epoch to 66.45% by the end.

### Analysis

- Gradual pruning exposed the accuracy cost of moving toward high sparsity.
- Mask enforcement prevented pruned weights from becoming nonzero again during optimizer updates.
- Fine-tuning recovered approximately 7.18 percentage points from the accuracy immediately after reaching the 70% target.
- Recovery was partial: final accuracy remained below the unpruned teacher baseline.
- This experiment was not a controlled comparison against a separately trained one-shot-pruned-and-fine-tuned model.

---

## 7. Experiment 3 — Symmetric weight quantization

A custom symmetric quantization implementation was developed using signed 8-bit integer values.

For a tensor with maximum absolute value \(\alpha\), the scale is:

\[
s = \frac{\alpha}{127}
\]

Quantization and reconstruction:

\[
q = \operatorname{clip}\left(\operatorname{round}\left(\frac{w}{s}\right), -127, 127\right)
\]

\[
\hat{w} = q \cdot s
\]

The implementation returns INT8 quantized values and a scale, and reconstructs floating-point values for error analysis.

### Toy tensor sanity check

| Metric | Result |
|---|---:|
| Input tensor | [-0.8, -0.5, 0, 0.3, 0.8] |
| Quantized tensor | [-127, -79, 0, 48, 127] |
| MSE | ~0.000002 |
| MAE | ~0.000945 |
| SNR | ~51.62 dB |

### First convolution layer: per-tensor vs. per-channel

| Metric | Per-tensor | Per-channel |
|---|---:|---:|
| Quantization granularity | One scale for tensor | One scale per output channel |
| Quantized range observed | -127 to 74 | -127 to 127 |
| MSE | 9.31696 × 10⁻⁶ | 1.57765 × 10⁻⁶ |
| MAE | 0.0025421 | 0.00086617 |
| SNR | 41.8113 dB | 49.524 dB |

### Analysis

- Per-channel quantization improved the measured SNR by approximately 7.71 dB for the first convolution layer.
- Its MSE was approximately 5.9 times lower than per-tensor quantization in this measurement.
- Per-channel scales accommodate differences in weight ranges across output channels.
- These results characterize weight reconstruction error for the tested layer. They do not establish full-model INT8 inference performance.

---

## 8. Experiment 4 — Activation calibration and quantization error

Forward hooks were attached to convolutional and linear layers to collect activation minima and maxima from calibration data. For symmetric activation quantization:

\[
\alpha = \max(|x_{\min}|, |x_{\max}|)
\]

\[
s_x = \frac{\alpha}{127}
\]

One recorded calibration pass produced the following activation ranges:

| Layer | Minimum | Maximum | Approx. symmetric scale |
|---|---:|---:|---:|
| `features.0` | -1.131321 | 1.462326 | 0.011515 |
| `features.3` | -0.701944 | 0.836433 | 0.006586 |
| `features.6` | -0.450713 | 0.347249 | 0.003549 |
| `classifier` | -0.104546 | 0.101258 | 0.000823 |

A subsequent layer-wise quantization-error measurement, using a separately recorded set of activation ranges, produced:

| Layer | MSE | MAE | SNR |
|---|---:|---:|---:|
| `features.0` | 1.1130 × 10⁻⁵ | 0.0029 | 38.5943 dB |
| `features.3` | 4.9927 × 10⁻⁶ | 0.0019 | 36.0675 dB |
| `features.6` | 7.5915 × 10⁻⁷ | 0.0008 | 37.3691 dB |
| `classifier` | 3.4243 × 10⁻⁸ | 0.0002 | 47.9088 dB |

### Analysis and caveats

- Activation ranges were captured at the outputs of Conv2d/Linear layers, before subsequent operations such as ReLU or pooling. Negative values are therefore expected.
- The second convolution layer had the lowest measured SNR in this activation experiment; the classifier had the highest.
- The calibration ranges from different runs were not identical. The tables above are separate recorded measurements and should not be combined as if they came from one identical calibration pass.
- Calibration data should represent expected inference inputs. If inference activations exceed calibrated ranges, clipping can occur.
- This work characterizes activation ranges and quantization error. It does not yet implement or benchmark a complete end-to-end integer-only inference graph.

---

## 9. Experiment 5 — Knowledge distillation

A narrower student CNN was trained using the teacher's softened output probabilities.

The distillation objective combined hard-label cross-entropy and temperature-scaled KL divergence:

\[
L = \alpha L_{\mathrm{CE}} + (1-\alpha)T^2L_{\mathrm{KD}}
\]

Configuration used:

| Setting | Value |
|---|---:|
| Temperature \(T\) | 4.0 |
| CE weight \(\alpha\) | 0.5 |
| KD weight | 0.5 |
| Optimizer | Adam |
| Learning rate | 0.001 |
| Training epochs | 10 |

The teacher was kept in evaluation mode, its parameters were frozen, and teacher logits were computed without gradient tracking.

### Distilled student training log

| Epoch | Total loss | CE loss | KD loss | Test accuracy |
|---:|---:|---:|---:|---:|
| 1 | 3.0088 | 1.3290 | 4.6885 | 65.44% |
| 2 | 1.1472 | 0.7864 | 1.5080 | 71.23% |
| 3 | 0.8069 | 0.6875 | 0.9262 | 74.41% |
| 4 | 0.6646 | 0.6395 | 0.6897 | 75.78% |
| 5 | 0.5848 | 0.6072 | 0.5623 | 75.80% |
| 6 | 0.5189 | 0.5757 | 0.4621 | 78.46% |
| 7 | 0.4704 | 0.5488 | 0.3920 | 79.16% |
| 8 | 0.4309 | 0.5266 | 0.3352 | 80.26% |
| 9 | 0.3981 | 0.5071 | 0.2891 | 80.88% |
| 10 | 0.3725 | 0.4911 | 0.2539 | 81.19% |

### Student architecture size

| Metric | Teacher | Student |
|---|---:|---:|
| Parameters | 93,962 | 23,946 |
| FP32 model size | 0.35844 MB | 0.09135 MB |
| Parameter reduction | — | 74.51% |
| Model-size reduction | — | 74.51% |
| Teacher-to-student size ratio | — | ~3.92× smaller |

The size reduction is due to the narrower student architecture, not distillation itself.

### Distillation vs. CE-only student

The same student architecture was also trained using ordinary cross-entropy to establish a comparison.

| Training approach | Test accuracy after 10 epochs |
|---|---:|
| Student trained with knowledge distillation | 81.19% |
| Student trained with CE only | 82.85% |
| CE-only difference | +1.66 percentage points |

### Analysis

- The distilled student reached 81.19% test accuracy with approximately one-quarter of the teacher's parameters.
- The CE-only student reached 82.85%, outperforming the current distillation run by 1.66 percentage points after 10 epochs.
- Therefore, the current experiment does not demonstrate an accuracy advantage from distillation under the selected configuration and training duration.
- This is a comparison of reported runs, not a multi-seed statistical study. Random seeds and training variation may affect the result.

---

## 10. Experiment 6 — Cubic pruning and masked fine-tuning on the distilled student

The pruning techniques developed earlier were reused on the distilled student model. A cubic sparsity schedule was used to reach a 30% target, followed by masked fine-tuning.

### Reported training results

| Epoch | Target sparsity | Threshold | Training loss | Test accuracy |
|---:|---:|---:|---:|---:|
| 1 | 0.00% | 0.0000 | 0.4468 | 82.87% |
| 2 | 26.25% | 0.0323 | 0.4272 | 83.96% |
| 3 | 30.00% | 0.0660 | 0.4184 | 84.02% |
| 4 | 30.00% | — | 0.4046 | 84.56% |
| 5 | 30.00% | — | 0.3954 | 84.60% |
| 6 | 30.00% | — | 0.3865 | 85.52% |
| 7 | 30.00% | — | 0.3804 | 85.45% |
| 8 | 30.00% | — | 0.3765 | 86.10% |
| 9 | 30.00% | — | 0.3663 | 86.32% |
| 10 | 30.00% | — | 0.3601 | 86.53% |

### Analysis

- The reported test accuracy increased from 82.87% at the first epoch to 86.53% at the final epoch.
- The final reported accuracy is higher than the original teacher baseline of 85.38%.
- The result combines continued training, gradual pruning, and masked fine-tuning. It does not isolate the contribution of pruning alone.
- **30% is the target sparsity. Actual final weight sparsity still needs to be measured and recorded.**
- The final pruned checkpoint's model size and latency have not yet been reported. Because the model retains dense tensor shapes, dense FP32 storage is expected to remain approximately unchanged unless sparse storage is used.

---

## 11. Results at a glance

| Experiment / model | Parameters | FP32 size | Test accuracy | Sparsity / precision | Notes |
|---|---:|---:|---:|---|---|
| Teacher baseline | 93,962 | 0.35844 MB | 85.38% | Dense FP32 | T4 mean latency 0.33189 ms |
| Student, CE-only | 23,946 | 0.09135 MB | 82.85% | Dense FP32 | 3.92× smaller than teacher |
| Student, distilled | 23,946 | 0.09135 MB | 81.19% | Dense FP32 | Current KD run |
| Teacher, one-shot 50% pruning | 93,962 | 0.35844 MB | 85.37% | 49.88% observed | Dense size unchanged |
| Teacher, one-shot 70% pruning | 93,962 | 0.35844 MB | 78.66% | 69.83% observed | Accuracy loss becomes material |
| Teacher, gradual 70% pruning + fine-tuning | 93,962 | Not separately remeasured | 66.45% | 70% target | Partial recovery after pruning |
| Distilled student, gradual 30% pruning + fine-tuning | 23,946 | Not yet remeasured | 86.53% | 30% target; actual pending | Latest reported training result |

**Important:** Values in this table come from different experiment runs and checkpoints. They are not all a single controlled leaderboard. A final comparison should rerun the chosen checkpoints through the same benchmark script and hardware configuration.

---

## 12. Key findings

1. **Architecture reduction gave the clearest measured dense-model size reduction.** The student uses 74.51% fewer parameters and has a 3.92× smaller FP32 footprint than the teacher.
2. **Magnitude pruning can preserve accuracy at moderate sparsity in this task.** One-shot teacher pruning remained close to baseline through 50% observed sparsity, while accuracy fell sharply at higher sparsities.
3. **Sparsity is not the same as compression in bytes or speed.** Dense tensors still store zero values, and ordinary dense kernels may not run faster.
4. **Fine-tuning matters after pruning.** The 70% teacher experiment showed partial accuracy recovery when training continued with fixed masks.
5. **Per-channel quantization reconstructed the tested convolution weights more accurately** than per-tensor quantization, with 49.52 dB versus 41.81 dB SNR.
6. **Knowledge distillation produced a compact student, but did not outperform CE-only training in the current 10-epoch comparison.**
7. **The latest distilled-student pruning experiment reached 86.53% reported accuracy at a 30% target sparsity.** Actual sparsity and standardized final benchmark metrics remain to be verified.
8. **Hardware-aware measurement is essential.** Model size, accuracy, latency distribution, and peak memory should be measured together.

---

## 13. Project status and next steps

| Area | Status |
|---|---|
| Fashion-MNIST teacher training | Completed |
| Baseline accuracy, size, latency and memory | Completed for teacher baseline |
| Unified benchmark modules | Implemented |
| One-shot global magnitude pruning | Completed |
| Cubic sparsity scheduling and masked fine-tuning | Completed on teacher |
| Symmetric weight quantization | Implemented and characterized |
| Activation calibration and error analysis | Implemented and characterized |
| Knowledge distillation | Implemented |
| CE-only student comparison | Completed |
| Cubic pruning + fine-tuning on distilled student | Training completed; final actual sparsity and benchmark pending |
| End-to-end INT8 inference | Not yet completed |
| Pruned + quantized student comparison | Not yet completed |
| Mobile/phone deployment benchmark | Not yet completed |
| Final budget search and deployment validation | Pending |

### Recommended next experiments

- Verify actual zero-weight sparsity in the final distilled-and-pruned student checkpoint.
- Run the unified benchmark on the CE-only student, distilled student, and distilled-plus-pruned student under identical settings.
- Implement a real INT8 inference path using a supported PyTorch quantization/runtime workflow, then measure accuracy, serialized size, and latency.
- Evaluate pruning plus quantization as a composed pipeline and compare it against each individual technique.
- Benchmark on the intended phone or edge target before making mobile FPS or latency claims.
- If the deployment target is 30 FPS, validate that end-to-end inference—including preprocessing and postprocessing—fits within the 33.33 ms/frame budget.

---

## 14. Reproducibility and reporting notes

- Keep an unmodified baseline checkpoint for every model family.
- Start each independent compression experiment from the intended baseline checkpoint; do not accidentally reuse a previously compressed checkpoint.
- Record random seeds, optimizer settings, data split, training epochs, batch size, hardware, and software versions.
- Report **target sparsity and measured actual sparsity** separately.
- Distinguish dense FP32 file size from sparse-encoded size and from INT8 serialized size.
- Distinguish quantization reconstruction experiments from actual integer inference.
- Use the same benchmark settings for all model comparisons.
- Report mean and tail latency (P95/P99), not only average FPS.
- Do not compare T4 measurements directly with mobile-device performance.

---

## 15. Repository structure

The project has used the following organization (individual filenames may evolve as experiments are consolidated):

```text
.
├── models/
│   ├── teacher.py
│   └── student.py
├── compression/
│   ├── pruning.py
│   └── quantization.py
├── benchmarks/
│   ├── accuracy.py
│   ├── size.py
│   ├── latency.py
│   ├── sparsity.py
│   ├── precision.py
│   ├── memory.py
│   ├── benchmark.py
│   └── run_baseline.py
├── experiments/
├── saved_model/
└── README.md
```

---

## 16. Summary

This project implements a practical model-compression workflow in PyTorch, from teacher-model training and baseline benchmarking through pruning, quantization analysis, activation calibration, and knowledge distillation.

The most concrete measured dense-model reduction so far comes from the narrower student architecture: **93,962 → 23,946 parameters, reducing the FP32 model footprint from 0.35844 MB to 0.09135 MB (approximately 74.51%).** Quantization experiments show that per-channel scaling can improve reconstruction quality, while pruning experiments reveal the trade-off between sparsity and predictive accuracy.

The latest distilled-student pruning run reports **86.53% test accuracy at a 30% target sparsity**, pending verification of actual sparsity and standardized checkpoint benchmarking. Full INT8 inference and mobile deployment remain future work.

**The central lesson:** model compression is not a single technique or a single metric. It is a measured trade-off among accuracy, model representation, memory, latency, and the capabilities of the target hardware.
