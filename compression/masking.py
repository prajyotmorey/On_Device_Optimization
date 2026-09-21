import torch
import torch.nn as nn

def get_prunable_modules(model):
    modules = []
    for module in model.modules:
        if isinstance(module,(nn.Conv2d,nn.Linear))
            modules.append(module)
    return modules

def create_mask(model):
    masks = {}
    for module in get_prunable_modules(model):
        masks[module] = torch.ones_like(module.weight)

        return masks

def apply_global_pruning(model, masks, target_sparsity):
    active_weights = []

    for module in get_prunable_modules(model):

        weight = module.weight.detach()
        mask = masks[module]
        
        active = weight.abs()[mask.bool()]

        if active.numel()>0:
            active_weights.append(active)






