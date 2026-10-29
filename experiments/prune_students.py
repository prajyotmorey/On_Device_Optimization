import torch
import torch.nn as nn
from models.student import StudentCNN
from training.dataloader import get_dataloaders
from compression.masking import apply_global_pruning, create_mask, apply_mask
from compression.cubic_sparsity import cubic_sparsity_schedule
from benchmarks.accuracy import evaluate_accuracy

# TARGET_SPARSITY = 0.3
# EPOCHS = 10
# LR = 1e-4

device = "cuda" if torch.cuda.is_available() else "cpu"
model = StudentCNN().to(device)
state = torch.load("saved_model/student_CE.pth", map_location=device,weights_only=True)
model.load_state_dict(state)

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
        
torch.save(model.state_dict(), "saved_model/pruned_finetuned_student_distilled.pth")