import torch
import torch.nn.functional as F


def distillation_loss(student_logits,teacher_logits,labels,temperature=4.0,alpha=0.5):
    # Standard supervised loss
    ce_loss = F.cross_entropy(student_logits,labels)

    # Teacher and student softened distributions
    teacher_probs = F.softmax(teacher_logits / temperature,dim=1)

    student_log_probs = F.log_softmax(student_logits / temperature,dim=1)

    # Teacher distribution is the target
    kd_loss = F.kl_div(student_log_probs,teacher_probs,reduction="batchmean") * (temperature ** 2)

    total_loss = (alpha * ce_loss+ (1 - alpha) * kd_loss)

    return total_loss, ce_loss, kd_loss