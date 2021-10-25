# # -*- coding:utf-8 -*-
from sch_test_frame.ut import OpUT
import numpy as np
import math
import warnings

from te import tvm
import te.lang.cce as tbe
from tbe.common.testing.testing import debug

warnings.filterwarnings("ignore")
ut_case = OpUT("dim_cpu", "dsl_cpu.test_dim_cpu_impl")


def _four2five(input_data, shape_info):
    n, c, h, w, c0 = shape_info[0], shape_info[1], shape_info[2], shape_info[3], 16
    c1 = int(math.ceil(c / 16.0))
    output = np.zeros((n * c1 * h * w * c0), input_data.dtype)
    for nIdx in range(n):
        n_factor = nIdx * c1 * h * w * c0
        for c1_index in range(c1):
            c1_factor = n_factor + c1_index * h * w * c0
            for hIdx in range(h):
                h_factor = c1_factor + hIdx * w * c0
                for wIdx in range(w):
                    w_factor = h_factor + wIdx * c0
                    for c0_index in range(c0):
                        idx = w_factor + c0_index
                        c_index = c0_index + c1_index * c0
                        if c_index < c:
                            output[idx] = input_data[nIdx, c0_index + c1_index * c0, hIdx, wIdx]
    return output.reshape(n, c1, h, w, c0)


def _five2four(input_data, shape_info):
    n, c, h, w, c0 = shape_info[0], shape_info[1], shape_info[2], shape_info[3], 16
    output = np.zeros((n * c * h * w), input_data.dtype)
    for nIdx in range(n):
        n_factor = nIdx * c * h * w
        for c_index in range(c):
            c_factor = n_factor + c_index * h * w
            for hIdx in range(h):
                h_factor = c_factor + hIdx * w
                for wIdx in range(w):
                    w_factor = h_factor + wIdx
                    c1_index = int(c_index / c0)
                    c0_index = c_index % c0
                    if c_index < c:
                        output[w_factor] = input_data[nIdx, c1_index, hIdx, wIdx, c0_index]
    return output.reshape(n, c, h, w)


def test_four_dim_to_five_dim(_):
    """
    convert NCHW to NC1HWC0
    @return: Ture && false
    """
    with debug():
        shape4d = (1, 32, 16, 128)
        shape5d = (1, 2, 16, 128, 16)
        input1 = tvm.placeholder(shape4d, name="input1", dtype="float16")
        output = tbe.compute_four2five(input1, shape4d)
        sch = tvm.create_schedule(output.op)
        func = tvm.build(sch, [input1, output], "c", "llvm", name="func")
        ctx = tvm.cpu(0)
        # 1. prepare kernel parameter
        a = tvm.nd.array(np.random.uniform(size=shape4d).astype(input1.dtype), ctx)
        b = tvm.nd.array(np.zeros(shape5d, dtype=output.dtype), ctx)
        # 2. run tbe kernel
        func(a, b)
        # 3.verify the correctness of output
        try:
            tvm.testing.assert_allclose(b.asnumpy(), _four2five(a.asnumpy(), shape4d))
        except AssertionError as e:
            print(e)
            return False
        return True


def test_five_dim_to_four_dim(_):
    """
    convert NC1HWC0 to NCHW
    @return: Ture && false
    """
    with debug():
        shape4d = (1, 32, 16, 128)
        shape5d = (1, 2, 16, 128, 16)
        input1 = tvm.placeholder(shape5d, name="input1", dtype="float16")
        output = tbe.compute_five2four(input1, shape4d)
        sch = tvm.create_schedule(output.op)
        func = tvm.build(sch, [input1, output], "c", "llvm", name="func")
        ctx = tvm.cpu(0)
        # 1. prepare kernel parameter
        a = tvm.nd.array(np.random.uniform(size=shape5d).astype(input1.dtype), ctx)
        b = tvm.nd.array(np.zeros(shape4d, dtype=output.dtype), ctx)
        # 2. run tbe kernel
        func(a, b)
        # 3.verify the correctness of output
        try:
            tvm.testing.assert_allclose(b.asnumpy(), _five2four(a.asnumpy(), shape4d))
        except AssertionError as e:
            print(e)
            return False
        return True


test_func_list = [
    test_four_dim_to_five_dim,
    test_five_dim_to_four_dim,
]
for item in test_func_list:
    ut_case.add_cust_test_func(test_func=item)


if __name__ == '__main__':
    import os
    from pathlib import Path

    _ASCEND_TOOLCHAIN_PATH_ENV = "TOOLCHAIN_HOME"
    simulator_lib_path = Path(os.environ.get(_ASCEND_TOOLCHAIN_PATH_ENV,
                                             "/usr/local/Ascend/toolkit")).joinpath("tools/simulator")
    ut_case.run(["Ascend310", "Ascend910A"], simulator_mode="pv", simulator_lib_path=simulator_lib_path)
