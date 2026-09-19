import random
import numpy as np
import torch

from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from models.teacher import TeacherCNN
from benchmark.accuracy import evaluate_accuracy

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def main():
    set_seed(42)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    transform = transforms.ToTensor()

    train_dataset = datasets.FashionMNIST(root="./data", train=True, download=True, transform=transform)

    test_dataset = datasets.FashionMNIST(root="./data", train=False, download=True, transform=transform)

    train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False)

    model = TeacherCNN(num_classes=10).to(device)

    criterion = torch.nn.CrossEntropyLoss()

    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)

    num_epochs = 10

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0

        for images, labels in train_loader:
            print("Training on batch...")
            images = images.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()

            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()

            optimizer.step()

            running_loss += loss.item()

        accuracy = evaluate_accuracy(model,test_loader,device)

        print(
                f"Epoch [{epoch + 1}/{num_epochs}], Loss: {running_loss / len(train_loader):.4f}, Accuracy: {accuracy:.4f}"
            )

    torch.save(model.state_dict(), "teacher_model.pth")

if __name__ == "__main__":
    main()







