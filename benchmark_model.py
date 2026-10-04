import torch

from benchmark_analysis import benchmark_model
from models.teacher import TeacherCNN
from results.update_csv import update_results_csv
from training.dataloader import get_dataloaders
from compression.pruning import apply_global_magnitude_pruning
from benchmarks.sparsity import  calculate_sparsity, layerwise_sparsity
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ##Dataset
    train_loader, test_loader = get_dataloaders()


    ## Model


    ##Apply pruning:
    target_sparsity=[x/100 for x in range(10,100,10)]
    prefix = 'teacher_pruned_'
    for i in target_sparsity:
        model = TeacherCNN()
        checkpoint = torch.load("teacher_model.pth", map_location=device)
        model.load_state_dict(checkpoint)
        model.to(device)
        print("**"*30)
        print("Running For Sparity: ",i)
        masks, threshold = apply_global_magnitude_pruning(model,i)
        print(f"Threshold: {threshold:.4f}")
        teacher_name = f"{prefix}{i}"
        ##check Sparsity
        sparsity = calculate_sparsity(model)
        layer_sparse = layerwise_sparsity(model)
        print(layer_sparse)
        ## Benchmarking
        results = benchmark_model(model, test_loader, device, teacher_name)    

        for key, value in results.items():
            print(f"{key}: {value}")
        
        update_results_csv(results)


 #Old Code for Only 50% Sparsity       
# import torch

# from benchmark_analysis import benchmark_model
# from models.teacher import TeacherCNN
# from results.update_csv import update_results_csv
# from training.dataloader import get_dataloaders
# from compression.pruning import apply_global_magnitude_pruning

# if __name__ == "__main__":
#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#     ##Dataset
#     train_loader, test_loader = get_dataloaders()


#     ## Model
#     model = TeacherCNN()
#     checkpoint = torch.load("teacher_model.pth", map_location=device)
#     model.load_state_dict(checkpoint)
#     model.to(device)

#     # ##Apply pruning:
#     # masks, threshold = apply_global_magnitude_pruning(model,target_sparsity=0.50)
#     # print(f"Threshold: {threshold:.4f}")

#     ## Benchmarking
#     results = benchmark_model(model, test_loader, device, "teacher_pruned_50")    

#     for key, value in results.items():
#         print(f"{key}: {value}")

#     update_results_csv(results)
