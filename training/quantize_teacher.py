import torch
from models.teacher import TeacherCNN
from compression.quantization import symmetric_quantize, symmetric_dequantize, quantization_error, calculate_SNR


if __name__ == "__main__":

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = TeacherCNN().to(device)
    checkpoint = torch.load("saved_model/teacher_model.pth", map_location=device)
    model.load_state_dict(checkpoint)

    model.eval()

    weight = model.features[0].weight.detach()

    #Quantize this weight for test:
    quantized_weight, scale = symmetric_quantize(weight)
    weight_reconstructed = symmetric_dequantize(quantized_weight, scale)

    snr = calculate_SNR(
    weight,
    weight_reconstructed
    )

    mse, mae = quantization_error(
    weight,
    weight_reconstructed
    )

    print("Original_weight dtype: ", weight.dtype)
    print("Original_weight range: ", (weight.min().item(), weight.max().item()))
    print("Scale: ", scale)
    print("*"*10,"After Quantize ","*"*10)
    print("Quantized dtype: ", quantized_weight.dtype)
    print("Quantized range: ", (quantized_weight.min().item(), quantized_weight.max().item()))
    print("*"*10,"After Dequantize ","*"*10)
    print("Reconstructed dtype: ", weight_reconstructed.dtype)
    print("Reconstructed range: ", (weight_reconstructed.min().item(), weight_reconstructed.max().item()))

    
    print("MSE:", mse.item())
    print("MAE:", mae.item())
    print("SNR:", snr, "dB")
