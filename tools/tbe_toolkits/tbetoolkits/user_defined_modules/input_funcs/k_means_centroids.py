#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
"""KMeans Input Generator"""
# Standard Packages
from typing import Tuple
from typing import Optional

# Third-Party Packages
import numpy as np
from .registry import register_input


@register_input(["k_means_centroids"])
def _k_means_input(_,
                   __,
                   input_arrays,
                   other_runtime_params,
                   ___,
                   ____) -> Tuple[Tuple[Optional[np.ndarray], ...],
                                  Tuple[Optional[np.ndarray], ...]]:

    use_actual_distance = other_runtime_params.get("use_actual_distance", False)

    ipt_0 = input_arrays[0]
    ipt_1 = input_arrays[1]
    ipt_2 = np.sum(ipt_0*ipt_0, axis=1, keepdims=True).astype("float32")
    ipt_3 = np.sum(ipt_1*ipt_1, axis=1, keepdims=True).T.astype("float32")

    if use_actual_distance:
        return (ipt_0, ipt_1, ipt_3, ipt_2), (ipt_0, ipt_1, ipt_3, ipt_2)
    return (ipt_0, ipt_1, ipt_3, None), (ipt_0, ipt_1, ipt_3, None)
