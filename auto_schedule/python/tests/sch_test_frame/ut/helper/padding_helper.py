from functools import reduce
from typing import Iterable

import numpy as np
import tbe.common.platform as platform
from tbe import tvm
from tbe.dsl.base import var_api
from tbe.dsl.base.padding import util as p_util
from tbe.dsl.base.padding.padding import Action
from tbe.tvm.tensor import Tensor


class SocVersionContext:
    def __init__(self, full_soc_version) -> None:
        self._pre_full_soc_version = platform.get_soc_spec("FULL_SOC_VERSION")
        self._full_soc_version = full_soc_version

    def __enter__(self):
        platform.set_current_compile_soc_info(self._full_soc_version)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        platform.set_current_compile_soc_info(self._pre_full_soc_version)


def padding_check(outs, actions):
    # type: (Iterable[Tensor], Iterable[Action]) -> bool
    tensor_action = {}
    for action in actions:
        tensor = action.get_tensor()
        actions = tensor_action.setdefault(tensor, [])
        actions.append(action)

    def dfs(tensor):
        # type: (Tensor) -> None
        if tensor in visited:
            return
        visited.add(tensor)

        for tensor_i in tensor.op.input_tensors:
            dfs(tensor_i)

        if not p_util.is_placeholder(tensor):
            pass

    visited = set()
    for out in outs:
        dfs(out)

    pass


def const_n(v):
    return var_api.const(v, "int32", {"axis_type": "N"})


def const_c1(v):
    return var_api.const(v, "int32", {"axis_type": "C1"})


def const_h(v):
    return var_api.const(v, "int32", {"axis_type": "H"})


def const_w(v):
    return var_api.const(v, "int32", {"axis_type": "W"})


def const_c0(v):
    return var_api.const(v, "int32", {"axis_type": "C0"})


def const_hw(v):
    return var_api.const(v, "int32", {"axis_type": ["H", "W"]})


def const_c(v):
    return var_api.const(v, "int32", {"axis_type": "C"})


def const_x(values, axis_types):
    const_values = []
    for v, axis_type in zip(values, axis_types):
        const_values.append(var_api.const(v, "int32", {"axis_type": axis_type}))
    return const_values


def soc_context(soc_version):
    return SocVersionContext(soc_version)


def soc_910():
    return SocVersionContext("Ascend910A")


def cmp_condition(shape, cond_func, expected_func):
    shape_int = tuple(i.value if isinstance(i, tvm.expr.IntImm) else i for i in shape)
    size = reduce(lambda x, y: x*y, shape_int)
    x1 = np.random.random(size).reshape(shape_int)

    cond_num, expected_num = [], []
    for i, _ in np.ndenumerate(x1):
        tvm_i = tuple(tvm.const(j) for j in i)
        if cond_func(*tvm_i).value == 1:
            cond_num.append(i)
        if expected_func(*i):
            expected_num.append(i)

    return cond_num == expected_num


def cmp_value(shape, value_func, expected_func):
    ph_0 = tvm.placeholder(shape, "float32", name="ph_0")
    shape_int = tuple(i.value if isinstance(i, tvm.expr.IntImm) else i for i in shape)
    size = reduce(lambda x, y: x*y, shape_int)
    x1 = np.random.random(size).reshape(shape_int)

    value_num, expected_num = [], []
    for i, _ in np.ndenumerate(x1):
        value_num.append(value_func(ph_0)(*i).indices)
        expected_num.append(expected_func(*i))

    return value_num == expected_num
