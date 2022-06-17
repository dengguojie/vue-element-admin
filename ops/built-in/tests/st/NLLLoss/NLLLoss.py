import torch

def nll_loss_compute(x, target, weight, y, total_weight,
                     reduction="mean", ignore_index=-100, kernel_name="nll_loss"):
    input_tensor = torch.from_numpy(x.get("value"))
    target = torch.from_numpy(target.get("value")).long()
    weight = torch.from_numpy(weight.get("value"))
    
    res = torch.nn.functional.nll_loss(input_tensor, target, weight=weight, size_average=None, 
                                       ignore_index=ignore_index, reduce=None, reduction=reduction)
    return [res.numpy()]
