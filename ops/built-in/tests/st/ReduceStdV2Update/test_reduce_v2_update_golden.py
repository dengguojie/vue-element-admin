import torch

# pylint: disable=unused-argument
def calc_expect_func(x, mean, output_var, dim, if_std, unbiased, keepdim):
    x = x["value"]
    x = torch.tensor(x)
    if(if_std):
        res = torch.std(x, dim, unbiased, keepdim).numpy()
        return[res, ]
    else:
        res = torch.var(x, dim, unbiased, keepdim).numpy()
        return [res, ]
