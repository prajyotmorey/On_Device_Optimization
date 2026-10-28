import torch

from models.teacher import TeacherCNN
from models.student import StudentCNN
from training.dataloader import get_dataloaders
from compression.distillation import distillation_loss
from benchmarks.accuracy import evaluate_accuracy


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    # Load the trained teacher
    teacher = TeacherCNN().to(device)

    checkpoint = torch.load(
        "saved_model/teacher_model.pth",
        map_location=device
    )

    teacher.load_state_dict(checkpoint)
    teacher.eval()

    # Freeze teacher parameters
    for param in teacher.parameters():
        param.requires_grad = False

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
        running_ce = 0.0
        running_kd = 0.0

        for inputs, labels in train_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)

            # Teacher generates targets; no gradients required
            with torch.no_grad():
                teacher_logits = teacher(inputs)

            student_logits = student(inputs)

            loss, ce_loss, kd_loss = distillation_loss(
                student_logits,
                teacher_logits,
                labels,
                temperature=temperature,
                alpha=alpha
            )

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            running_ce += ce_loss.item()
            running_kd += kd_loss.item()

        accuracy = evaluate_accuracy(
            student,
            test_loader,
            device
        )

        n = len(train_loader)

        print(
            f"Epoch [{epoch + 1}/{num_epochs}] "
            f"Loss={running_loss/n:.4f} "
            f"CE={running_ce/n:.4f} "
            f"KD={running_kd/n:.4f} "
            f"Test Accuracy={accuracy:.4f}"
        )

    torch.save(
        student.state_dict(),
        "saved_model/student_distilled.pth"
    )


if __name__ == "__main__":
    main()