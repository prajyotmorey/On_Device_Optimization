import torch

@torch.no_grad() #temporarily disable gradient calculation for inference


def evaluate_accuracy(model, dataloader, device):
    model.eval() #inference mode

    correct = 0
    total = 0

    for images, labels in dataloader:
        images = images.to(device)
        labels = labels.to(device)

        outputs = model(images)
        prediction = outputs.argmax(dim=1) #get the index of the max log-probability

        correct +=(prediction == labels).sum().item()
        total += labels.size(0)

    accuracy = correct / total
    return accuracy

