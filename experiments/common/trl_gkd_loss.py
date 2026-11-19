# Copied from TRL (Apache License 2.0): GKDTrainer.generalized_jsd_loss; see THIRD_PARTY_NOTICES.md.
import torch
import torch.nn.functional as F
SOURCE_MD5 = '45a12846c2867b30588ec023b5616c7f'
SOURCE_LINES = (238, 320)

def generalized_jsd_loss(student_logits, teacher_logits, labels=None, beta=0.5, temperature=1.0, reduction='batchmean', num_items_in_batch=None):
    student_logits = student_logits / temperature
    teacher_logits = teacher_logits / temperature
    student_log_probs = F.log_softmax(student_logits, dim=-1)
    teacher_log_probs = F.log_softmax(teacher_logits, dim=-1)
    if beta == 0:
        jsd = F.kl_div(student_log_probs, teacher_log_probs, reduction='none', log_target=True)
    elif beta == 1:
        jsd = F.kl_div(teacher_log_probs, student_log_probs, reduction='none', log_target=True)
    else:
        beta = torch.tensor(beta, dtype=student_log_probs.dtype, device=student_log_probs.device)
        mixture_log_probs = torch.logsumexp(torch.stack([student_log_probs + torch.log1p(-beta), teacher_log_probs + torch.log(beta)]), dim=0)
        kl_teacher = F.kl_div(mixture_log_probs, teacher_log_probs, reduction='none', log_target=True)
        kl_student = F.kl_div(mixture_log_probs, student_log_probs, reduction='none', log_target=True)
        jsd = beta * kl_teacher + (1 - beta) * kl_student
    if labels is not None:
        mask = labels != -100
        jsd = jsd[mask]
    if num_items_in_batch is not None:
        jsd_sum = jsd.sum()
        if isinstance(num_items_in_batch, torch.Tensor):
            num_items_in_batch = num_items_in_batch.to(jsd_sum.device)
        return jsd_sum / num_items_in_batch
    if reduction == 'batchmean':
        denom = mask.sum().clamp_min(1) if labels is not None else max(jsd.size(0), 1)
        return jsd.sum() / denom
    elif reduction == 'sum':
        return jsd.sum()
    elif reduction == 'mean':
        return jsd.mean()
