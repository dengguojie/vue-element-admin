import numpy as np


def nll_loss_grad_compute(x,
                  y_grad,
                  target,
                  weight,
                  total_weight,
                  x_grad,
                  reduction="mean",
                  ignore_index=-100,
                  kernel_name="nll_loss_grad"):

    x_shape = x.get("shape")
    if len(x_shape) == 1:
        x_shape  [1, x_shape[0]]
    x_dtype = x.get("dtype")
    y_grad_value = y_grad.get("value")
    target_value = target.get("value")
    weight_value = weight.get("value")
    total_weight_value = total_weight.get("value")
    x_grad_shape = x_shape

    n_dim = x_shape[0]
    if x_dtype == "float":
        loss = np.zeros(x_shape).astype(np.float32)
    else:
        loss = np.zeros(x_shape).astype(x_dtype)

    for i in range(n_dim):
        if ((target_value[i] == ignore_index) and (ignore_index >= 0) and (ignore_index < x_shape[-1])) \
            or (target_value[i] < 0) or (target_value[i] >= x_shape[-1]):
            continue
        valid_weight = weight_value[target_value[i]]

        if reduction == "none":
            loss[i][target_value[i]] = -1 * y_grad_value[i] * valid_weight
        elif reduction == "sum":
            loss[i][target_value[i]] = -1 * y_grad_value[0] * valid_weight
        elif reduction == "mean":
            loss[i][target_value[i]] = -1 * y_grad_value[0] * valid_weight / total_weight_value[0]

    loss = loss.reshape(x_grad_shape)
    return loss

