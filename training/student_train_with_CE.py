import torch

from models.teacher import TeacherCNN
from models.student import StudentCNN
from training.dataloader import get_dataloaders
from compression.distillation import distillation_loss
from benchmarks.accuracy import evaluate_accuracy


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # Initialize student
    student = StudentCNN().to(device)

    train_loader, test_loader = get_dataloaders()

    optimizer = torch.optim.Adam(
        student.parameters(),
        lr=0.001
    )

    num_epochs = 10
    temperature = 4.0
    alpha = 0.5

    for epoch in range(num_epochs):
        student.train()

        running_loss = 0.0

        for inputs, labels in train_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)

        

            student_logits = student(inputs)



            loss = torch.nn.functional.cross_entropy(
                    student_logits,
                    labels
                    )

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        accuracy = evaluate_accuracy(
            student,
            test_loader,
            device
        )

        n = len(train_loader)

        print(
            f"Epoch [{epoch + 1}/{num_epochs}] "
            f"Loss={running_loss/n:.4f} "
            f"Test Accuracy={accuracy:.4f}"
        )

    torch.save(
        student.state_dict(),
        "saved_model/student_CE.pth"
    )


if __name__ == "__main__":
    main()