import torch
import numpy as np


def gather_elements_torch(x, index, y, dim=0):
    # pylint: disable=unused-argument
    """
    gather_elements inferface

    Parameters
    ----------
    x: input params value, shape, dtype and range
    index: input index value, shape, dtype and range
    y: output shape, dtype and range
    dim: which dim to gather on, attr

    Returns
    -------
    result of gather elements in pytorch
    """
    
    data_pt = torch.from_numpy(x["value"])
    index_pt = torch.from_numpy(index["value"].astype(np.int64))
    result = torch.gather(data_pt, dim, index_pt).numpy()
    return [result, ]