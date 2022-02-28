# # -*- coding:utf-8 -*-
from sch_test_frame.ut import OpUT
import numpy as np
import warnings

from te import tvm
import tbe

warnings.filterwarnings("ignore")
ut_case = OpUT("array_split", "split.test_dynamic_split_compute_impl")


def test_split_cpu_api_axis_zero(_):
    """
    for split api
    @return: Ture && false
    """
    m, n = 1024, 5
    size_splits = [512, 512]
    data = tvm.placeholder((m, n), name="data", dtype="float16")
    with tbe.common.context.op_context.OpContext("dynamic"):
        outputs = tbe.dsl.split(data, 0, size_splits)
    res_op = []
    for out in outputs:
        res_op.append(out.op)
    sch = tvm.create_schedule(res_op)
    func = tvm.build(sch, [data, *outputs], "c", "llvm", name="func")
    ctx = tvm.cpu(0)
    # 1. prepare kernel parameter
    a = tvm.nd.array(np.random.uniform(size=(m, n)).astype(data.dtype), ctx)
    outs = []
    for s_size in size_splits:
        outs.append(tvm.nd.array(np.random.uniform(size=(s_size, n)).astype(data.dtype), ctx))
    # 2. run tbe kernel
    func(a, *outs)
    # 3.verify the correctness of output
    try:
        last_index = 0
        input_data = a.asnumpy().reshape(m, n)
        for s_size, out in zip(size_splits, outs):
            tvm.testing.assert_allclose(out.asnumpy(), input_data[last_index:last_index + s_size, :])
            last_index += s_size
    except AssertionError as e:
        print(e)
        return False
    return True


def test_split_cpu_api_axis_one(_):
    """
    for split api
    @return: Ture && false
    """
    m, n = 5, 1024
    size_splits = [500, 524]
    data = tvm.placeholder((m, n), name="data", dtype="float32")
    with tbe.common.context.op_context.OpContext("dynamic"):
        outputs = tbe.dsl.split(data, 1, size_splits)
    res_op = []
    for out in outputs:
        res_op.append(out.op)
    sch = tvm.create_schedule(res_op)
    func = tvm.build(sch, [data, *outputs], "c", "llvm", name="func")
    ctx = tvm.cpu(0)
    # 1. prepare kernel parameter
    a = tvm.nd.array(np.random.uniform(size=(m, n)).astype(data.dtype), ctx)
    outs = []
    for s_size in size_splits:
        outs.append(tvm.nd.array(np.random.uniform(size=(m, s_size)).astype(data.dtype), ctx))
    # 2. run tbe kernel
    func(a, *outs)
    # 3.verify the correctness of output
    try:
        last_index = 0
        input_data = a.asnumpy().reshape(m, n)
        for s_size, out in zip(size_splits, outs):
            tvm.testing.assert_allclose(out.asnumpy(), input_data[:, last_index:last_index + s_size])
            last_index += s_size
    except AssertionError as e:
        print(e)
        return False
    return True


def test_split_cpu_api_dtype_is_uint64(_):
    """
    for split api
    @return: Ture && false
    """
    m, n = 5, 1024
    size_splits = [1, 1023]
    data = tvm.placeholder((m, n), name="data", dtype="uint64")
    with tbe.common.context.op_context.OpContext("dynamic"):
        outputs = tbe.dsl.split(data, 1, size_splits)
    res_op = []
    for out in outputs:
        res_op.append(out.op)
    sch = tvm.create_schedule(res_op)
    func = tvm.build(sch, [data, *outputs], "c", "llvm", name="func")
    ctx = tvm.cpu(0)
    # 1. prepare kernel parameter
    a = tvm.nd.array(np.random.uniform(size=(m, n)).astype(data.dtype), ctx)
    outs = []
    for s_size in size_splits:
        outs.append(tvm.nd.array(np.random.uniform(size=(m, s_size)).astype(data.dtype), ctx))
    # 2. run tbe kernel
    func(a, *outs)
    # 3.verify the correctness of output
    try:
        last_index = 0
        input_data = a.asnumpy().reshape(m, n)
        for s_size, out in zip(size_splits, outs):
            tvm.testing.assert_allclose(out.asnumpy(), input_data[:, last_index:last_index + s_size])
            last_index += s_size
    except AssertionError as e:
        print(e)
        return False
    return True


def test_split_cpu_api_three_dims(_):
    """
    for split api
    @return: Ture && false
    """
    m, n, q = 6, 1024, 34
    size_splits = [100, 200, 724]
    data = tvm.placeholder((m, n, q), name="data", dtype="float16")
    with tbe.common.context.op_context.OpContext("dynamic"):
        outputs = tbe.dsl.split(data, 1, size_splits)
    res_op = []
    for out in outputs:
        res_op.append(out.op)
    sch = tvm.create_schedule(res_op)
    func = tvm.build(sch, [data, *outputs], "c", "llvm", name="func")
    ctx = tvm.cpu(0)
    # 1. prepare kernel parameter
    a = tvm.nd.array(np.random.uniform(size=(m, n, q)).astype(data.dtype), ctx)
    outs = []
    for s_size in size_splits:
        outs.append(tvm.nd.array(np.random.uniform(size=(m, s_size, q)).astype(data.dtype), ctx))
    # 2. run tbe kernel
    func(a, *outs)
    # 3.verify the correctness of output
    try:
        last_index = 0
        input_data = a.asnumpy().reshape(m, n, q)
        for s_size, out in zip(size_splits, outs):
            tvm.testing.assert_allclose(out.asnumpy(), input_data[:, last_index:last_index + s_size, :])
            last_index += s_size
    except AssertionError as e:
        print(e)
        return False
    return True


test_func_list = [
    test_split_cpu_api_axis_zero,
    test_split_cpu_api_axis_one,
    test_split_cpu_api_dtype_is_uint64,
    test_split_cpu_api_three_dims,
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
