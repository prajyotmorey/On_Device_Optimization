'''
     Input: tensor - a PyTorch tensor to be quantized
     Result: quantized_tensor, dequantized_tensor and reconstructed tensor
'''
import torch


        #################################################
                  ##Per- Tensor Quantization##
        #################################################

def symmetric_quantize(tensor):
    max_val = tensor.abs().max()
    scale = max_val / 127.0  # For int8, the range is -127 to 127

    quantized_tensor = torch.round(tensor / scale)
    quantized_tensor = torch.clamp(quantized_tensor, -127, 127).to(torch.int8)
    return quantized_tensor, scale

def symmetric_dequantize(quantized_tensor, scale):
    dequantized_tensor = quantized_tensor.float() * scale
    return dequantized_tensor


def quantization_error(original_tensor, reconstructed_tensor):
    error = original_tensor - reconstructed_tensor
    mse = torch.mean(error **2 )
    mae = torch.mean(torch.abs(error))
    return mse, mae

def calculate_SNR(original_tensor, reconstructed_tensor):
    signal_power = torch.mean(original_tensor ** 2)
    noise_power = torch.mean((original_tensor - reconstructed_tensor) ** 2)
    snr = 10 * torch.log10(signal_power / noise_power)
    return snr



        #################################################
                  ##Per- Channel Quantization##
        #################################################
def per_channel_symmetric_quantize(tensor):
    # Assuming tensor shape is (out_channels, in_channels, height, width) for Conv2D weights
    out_channels = tensor.size(0)
    quantized_tensor = torch.zeros_like(tensor, dtype=torch.int8)
    scales = torch.zeros(out_channels)

    for i in range(out_channels):
        channel_weights = tensor[i]
        max_val = channel_weights.abs().max()
        scale = max_val / 127.0  # For int8
        scales[i] = scale

        quantized_channel = torch.round(channel_weights / scale)
        quantized_channel = torch.clamp(quantized_channel, -127, 127).to(torch.int8)
        quantized_tensor[i] = quantized_channel
        #print("Quantized Channel {}: dtype: {}, range: ({}, {})".format(i, quantized_channel.dtype, quantized_channel.min().item(), quantized_channel.max().item()))
    return quantized_tensor, scales 

def per_channel_symmetric_dequantize(quantized_tensor, scales):
    out_channels = quantized_tensor.size(0)
    dequantized_tensor = torch.zeros_like(quantized_tensor, dtype=torch.float32)

    for i in range(out_channels):
        scale = scales[i]
        dequantized_channel = quantized_tensor[i].float() * scale
        dequantized_tensor[i] = dequantized_channel

    return dequantized_tensor

# ##Test code







# if __name__ == "__main__":
#     x =  torch.tensor([-0.8,-0.5,0.0,0.3,0.8])
#     quantized_tensor, scale = symmetric_quantize(x)
#     x_hat = symmetric_dequantize(quantized_tensor, scale)

#     print("Original Tensor: ", x)
#     print("Quantized Tensor: ", quantized_tensor)
#     print("Scale: ", scale)
#     print("Reconstructed Tensor: ", x_hat)

#     mse, mae = quantization_error(x, x_hat)
#     print(f"Mean Squared Error: {mse :.6f}, Mean Absolute Error: {mae.item():.6f}")

#     snr = calculate_SNR(x, x_hat)
#     print(f"Signal-to-Noise Ratio (SNR): {snr.item():.2f} dB")

