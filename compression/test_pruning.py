import torch

from compression.pruning import apply_global_magnitude_pruning, collect_weights
from models.teacher import TeacherCNN
from benchmarks.sparsity import calculate_sparsity
import benchmark_model 


# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# model = TeacherCNN()

# checkpoint = torch.load("teacher_model.pth", map_location=device)
# model.load_state_dict(checkpoint)
# model = model.to(device)
# model.eval()

def test_pruning(model,target_sparsity=0.50):
    mask, threshold = apply_global_magnitude_pruning(model, target_sparsity=0.50)
    #print(threshold)
    sparsity = calculate_sparsity(model)

    print(f"Sparsity:{sparsity *100:.2f}%")



