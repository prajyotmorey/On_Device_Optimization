import torch
import torch.nn as nn

def get_prunable_parameters(model):
    """
    Get the prunable parameters of a model.

    Args:
        model (nn.Module): The PyTorch model.

    Returns:
        list: A list of prunable parameters.
    """
    prunable_params = []
    for name, module in model.named_modules():
        if isinstance(module, (nn.Conv2d, nn.Linear)):
            prunable_params.append((module.weight))
    return prunable_params

def collect_weights(model):
    """
    Collect the weights of a model.

    Args:
        model (nn.Module): The PyTorch model.

    Returns:
        torch.Tensor: A flattened tensor containing the weights of the model.
    """
    parameters = get_prunable_parameters(model)
    weights = []
    for parameter in parameters:
        weights.append(parameter.detach().abs().flatten()) # Remove computation graph, take magnitude and flatten
    return torch.cat(weights)



def apply_global_magnitude_pruning(
    model,
    target_sparsity
):

    parameters = get_prunable_parameters(model)

    all_weights = collect_weights(model)

    threshold = torch.quantile(
        all_weights,
        target_sparsity
    )

    masks = {}

    for parameter in parameters:

        mask = (
            parameter.detach().abs()
            > threshold
        ).float()

        parameter.data.mul_(mask)

        masks[id(parameter)] = mask

    return masks, threshold