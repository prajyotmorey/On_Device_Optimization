
import torch
from models.teacher import TeacherCNN
from training.dataloader import get_dataloaders
from compression.masking import apply_global_pruning, create_mask, apply_mask
from compression.cubic_sparsity import cubic_sparsity_schedule
from benchmarks.accuracy import evaluate_accuracy

if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model =TeacherCNN()
    checkpoint = torch.load("saved_model/teacher_model.pth", map_location=device)
    model.load_state_dict(checkpoint)
    model.to(device)

    mask = create_mask(model)

    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = torch.nn.CrossEntropyLoss()
    train_loader, test_loader = get_dataloaders()
    for epoch in range(10):
        if epoch < 7: # Prunning phase, only till 70% sparsity
            
            target_sparsity = cubic_sparsity_schedule(epoch, 6, 0.70)

            mask, threshold = apply_global_pruning(model, mask, target_sparsity)

            print(f"Epoch: {epoch}, Target Sparsity: {target_sparsity:.4f}, Threshold: {threshold:.4f}")

        running_loss = 0.0
        model.train()

        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            apply_mask(model, mask)
        accuracy = evaluate_accuracy(model,test_loader,device)
        print(f"Epoch [{epoch + 1}/10], Loss: {running_loss / len(train_loader):.4f}, Accuracy: {accuracy:.4f}")

    torch.save(model.state_dict(), "saved_model/pruned_finetuned_model.pth")