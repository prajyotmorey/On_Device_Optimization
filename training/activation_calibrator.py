import torch
import torch.nn as nn
from models.teacher import TeacherCNN
from training.dataloader import get_dataloaders
from compression.quantization import calibrated_symmetric_quantize, calibrated_symmetric_dequantize, quantization_error, calculate_SNR
from compression.calibration import ActivationCalibrator

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
    print("*"*100," Layer Wise Min-Max Calculation ","*"*100)
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

    print("*"*100," Layer Wise Scale Calculation ","*"*100)
    for name, (min_val, max_val) in calibration_ranges.items():
        alpha = max(abs(min_val), abs(max_val))
        scales[name] = max(alpha / 127.0, 1e-8)
        print(f"{name}: scale={scales[name]:.8f}")

    for name, activation in calibrator.activations.items():
        activation_0 = calibrator.activations[name]
        scale = scales[name]
        ##Quantization
        quantized_activation = calibrated_symmetric_quantize(activation_0,scale)
        reconstructed_activation = calibrated_symmetric_dequantize(quantized_activation,scale)

        quantization_error_mse, quantization_error_mae = quantization_error(activation_0, reconstructed_activation)
        quantization_snr = calculate_SNR(activation_0,reconstructed_activation)

        print("*"*100)
        print("Quantization Of Layer_",name)
        print("Original Activation dtype: ", activation_0.dtype)
        print("Quantized dtype: ",quantized_activation.dtype)
        print("MSE: ",quantization_error_mse)
        print("MAE: ",quantization_error_mae)
        print("SNR: ",quantization_snr,"dB")
        
