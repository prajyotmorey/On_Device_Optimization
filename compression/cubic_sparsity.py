def cubic_sparsity_schedule(step, total_steps, final_sparsity):

    progress = step / total_steps

    progress = min(max(progress,0.0),1.0)

    sparsity = (final_sparsity  * (1-(1-progress) ** 3))

    return sparsity



# #Test code
# for step in range(8):
#     s = cubic_sparsity_schedule(step, 7, 0.70)
#     #print(f"Step: {step}, Sparsity: {s:.4f}")
    