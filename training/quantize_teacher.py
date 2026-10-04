import torch
from models.teacher import TeacherCNN
from compression.quantization import symmetric_quantize, symmetric_dequantize, quantization_error, calculate_SNR, per_channel_symmetric_quantize, per_channel_symmetric_dequantize  

def per_tensor_quantization_of_layer(layer_weight):
    quantized_weight, scale = symmetric_quantize(layer_weight)
    weight_reconstructed = symmetric_dequantize(quantized_weight, scale)

    snr = calculate_SNR(
    layer_weight,
    weight_reconstructed
    )

    mse, mae = quantization_error(
    layer_weight,
    weight_reconstructed
    )

    print("*"*10,"Per Tensor Quantization Results","*"*10)
    print("*"*10,"Before Quantization ","*"*10)
    print("Original_weight dtype: ", layer_weight.dtype)
    print("Original_weight range: ", (layer_weight.min().item(), layer_weight.max().item()))
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

def per_channel_quantization_of_layer(layer_weight):
    quantized_weight, scales = per_channel_symmetric_quantize(layer_weight)
    weight_reconstructed = per_channel_symmetric_dequantize(quantized_weight, scales)

    snr = calculate_SNR(
    layer_weight,
    weight_reconstructed
    )

    mse, mae = quantization_error(
    layer_weight,
    weight_reconstructed
    )
    print("*"*10,"Per Channel Quantization Results","*"*10)
    print("*"*10,"Before Quantization ","*"*10)
    print("Original_weight dtype: ", layer_weight.dtype)
    print("Original_weight range: ", (layer_weight.min().item(), layer_weight.max().item()))
    print("Scales shape: ", scales.shape)
    print("*"*10,"After Quantize ","*"*10)
    print("Quantized dtype: ", quantized_weight.dtype)
    print("Quantized range: ", (quantized_weight.min().item(), quantized_weight.max().item()))
    print("*"*10,"After Dequantize ","*"*10)
    print("Reconstructed dtype: ", weight_reconstructed.dtype)
    print("Reconstructed range: ", (weight_reconstructed.min().item(), weight_reconstructed.max().item()))

       
    print("MSE:", mse.item())
    print("MAE:", mae.item())
    print("SNR:", snr, "dB")    

if __name__ == "__main__":

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = TeacherCNN().to(device)
    checkpoint = torch.load("saved_model/teacher_model.pth", map_location=device)
    model.load_state_dict(checkpoint)

    model.eval()

    weight = model.features[0].weight.detach()

    per_tensor_quantization_of_layer(weight)
    per_channel_quantization_of_layer(weight)
    