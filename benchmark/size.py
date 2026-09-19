import torch
from models.teacher import TeacherCNN

def count_parameters(model):
    return sum(p.numel() for p in model.parameters())

def count_trainable_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def model_size(model):
    total_bytes = 0
    for parameter in model.parameters():
        total_bytes += parameter.numel() * parameter.element_size()

        for buffer in model.buffers():
            total_bytes += buffer.numel() * buffer.element_size()   

    return total_bytes / (1024 * 1024)  # Convert bytes to megabytes


def model_report(model):
    parameters = count_parameters(model)
    trainable_parameters = count_trainable_parameters(model)
    size_mb = model_size(model)

    return {
        "parameters" : parameters,
        "trainable_parameters" : trainable_parameters,
        "size_mb" : size_mb
    }
