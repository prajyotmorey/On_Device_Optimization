import torch
import torch.nn as nn
from models.teacher import TeacherCNN
from training.dataloader import get_dataloaders

from compression.quantization import calibrated_symmetric_quantize, calibrated_symmetric_dequantize, quantization_error, calculate_SNR


class ActivationCalibrator:

    def __init__(self):
        self.activations = {}
        self.min_values = {}
        self.max_values = {}
        self.handles = []
        
    def hook_fn(self, name):

        def hook(module, inputs, output):
            self.activations[name] = output.detach().clone()
            current_min = output.detach().min().item()
            current_max = output.detach().max().item()

            if name not in self.min_values:

                self.min_values[name] = current_min
                self.max_values[name] = current_max

            else:

                self.min_values[name] = min(
                    self.min_values[name],
                    current_min
                )

                self.max_values[name] = max(
                    self.max_values[name],
                    current_max
                )

        return hook

    def register_hooks(self, model):

        for name, module in model.named_modules():

            if isinstance(module,(nn.Conv2d, nn.Linear)):

                handle = module.register_forward_hook(self.hook_fn(name))

                self.handles.append(handle)

    def remove_hooks(self):
        for handle in self.handles:
            handle.remove()
        
        self.handles = []

#Test Code
if __name__ == "__main__":
    model = TeacherCNN()
    train_loader, test_loader = get_dataloaders(batch_size=128)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()

    calibrator = ActivationCalibrator()

    calibrator.register_hooks(model)

    with torch.no_grad():

        for inputs, labels in train_loader:
            inputs = inputs.to(device)
            model(inputs)

    calibrator.remove_hooks()

    calibration_ranges={}
    for name in calibrator.min_values:
        calibration_ranges[name] = (calibrator.min_values[name], calibrator.max_values[name])

        print(
                name,
                "min =",
                calibrator.min_values[name],
                "max =",
                calibrator.max_values[name]
            )

        #Collect and Calculate Scales
        scales = {}

    for name, (min_val, max_val) in calibration_ranges.items():
        alpha = max(abs(min_val), abs(max_val))
        scales[name] = max(alpha / 127.0, 1e-8)
        print(f"{name}: scale={scales[name]:.8f}")

    activation_0 = calibrator.activations['features.0']
    scale = scales['features.0']
    ##Quantization
    quantized_activation = calibrated_symmetric_quantize(activation_0,scale)
    reconstructed_activation = calibrated_symmetric_dequantize(quantized_activation,scale)

    quantization_error_mse, quantization_error_mae = quantization_error(activation_0, reconstructed_activation)
    quantization_snr = calculate_SNR(activation_0,reconstructed_activation)

    print("*"*100)
    print("Quantization Of Layer_1: ")
    print("Original Activation dtype: ", activation_0.dtype)
    print("Quantized dtype: ",quantized_activation.dtype)
    print("MSE: ",quantization_error_mse)
    print("MAE: ",quantization_error_mae)
    print("SNR: ",quantization_snr,"dB")




    