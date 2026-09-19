import torch

def calculate_sparsity(model):
    total_params = 0
    zero_params = 0

    for param in model.parameters():
        total_params += param.numel()
        zero_params += torch.sum(param == 0).item()
    sparsity = zero_params / total_params if total_params > 0 else 0
    return sparsity


def layerwise_sparsity(model):
    layerwise_sparsity_dict = {}
    for name, param in model.named_parameters():
        total_params = param.numel()
        zero_params = torch.sum(param == 0).item()
        sparsity = zero_params / total_params if total_params > 0 else 0
        layerwise_sparsity_dict[name] = sparsity
    return layerwise_sparsity_dict

