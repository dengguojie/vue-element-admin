#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
"""
Davinci operator parameter transformation map
"""

# Third-party Packages
from . import op_tiling_trans_registry


def __eliminate_duplicate_axes(axis, input_tensor):
    axis = tuple(set([_ax if _ax >= 0 else len(input_tensor.shape) + _ax for _ax in axis]))
    return axis


@op_tiling_trans_registry.register_func(("OneHot",))
def _one_hot(input_shapes: tuple, params: dict):
    params = params.copy()
    params["axis"] = __eliminate_duplicate_axes(input_shapes[0], params["axis"])
    return input_shapes, params


@op_tiling_trans_registry.register_func(("Tile",))
def _one_hot(input_shapes: tuple, params: dict):
    params = params.copy()
    params["input_m"] = params["multiples"]
    return input_shapes, params


@op_tiling_trans_registry.register_func(("ReduceSum", "ReduceProd", "ReduceMax", "ReduceMin", "ReduceMean",
                                         "ReduceAny", "ReduceAll",
                                         "reduce_sum", "reduce_prod", "reduce_max", "reduce_min", "reduce_mean",
                                         "reduce_any", "reduce_all"))
def _one_hot(input_shapes: tuple, params: dict):
    params = params.copy()
    if "axis" in params and "axes" not in params:
        params["axes"] = params["axis"]
    return input_shapes, params
