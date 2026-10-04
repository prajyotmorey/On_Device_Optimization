import torch
import torch.nn as nn
from models.teacher import TeacherCNN

def get_prunable_modules(model):
    modules = []

    for module in model.modules():
        if isinstance(module,(nn.Conv2d,nn.Linear)):
            modules.append(module)

    return modules

def create_mask(model):
    masks = {}

    for module in get_prunable_modules(model):
        masks[module] = torch.ones_like(module.weight)
        #print(f"Mask created for module: {module.__class__.__name__}, shape: {masks[module].shape}")

    return masks

def apply_global_pruning(model, masks, target_sparsity):

    active_weights = []

    for module in get_prunable_modules(model):

        weight = module.weight.detach()
        mask = masks[module]

        active = weight.abs()[mask.bool()]

        if active.numel() > 0:
            active_weights.append(active)

    all_active_weights = torch.cat(active_weights)

    threshold = torch.quantile(
        all_active_weights,
        target_sparsity
    )

    for module in get_prunable_modules(model):

        weight = module.weight.detach()
        old_mask = masks[module]

        new_mask = old_mask * (
            weight.abs() > threshold
        ).float()

        masks[module] = new_mask

        with torch.no_grad():
            module.weight.mul_(new_mask)

    return masks, threshold

def apply_mask(model, masks):
    with torch.no_grad():
        for module in get_prunable_modules(model):
            module.weight.mul_(masks[module])



# #Test 
# model = TeacherCNN()
# mask = create_mask(model)
# print(mask)