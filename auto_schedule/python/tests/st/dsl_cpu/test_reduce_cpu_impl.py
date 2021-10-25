# # -*- coding:utf-8 -*-
from sch_test_frame.ut import OpUT
import numpy as np
import warnings

from te import tvm
import te.lang.cce as tbe

warnings.filterwarnings("ignore")
ut_case = OpUT("reduce_cpu", "dsl_cpu.test_reduce_cpu_impl")


def test_reduce_sum_cpu_api_float32(_):
    """
    for reduce_sum api
    @return: Ture && false
    """
    n = 1024
    input1 = tvm.placeholder((n, n), name="input1", dtype="float32")
    output = tbe.sum(input1, axis=1)
    sch = tvm.create_schedule(output.op)
    func = tvm.build(sch, [input1, output], "c", "llvm", name="func")
    ctx = tvm.cpu(0)
    # 1. prepare kernel parameter
    a = tvm.nd.array(np.random.uniform(size=(n, n)).astype(input1.dtype), ctx)
    b = tvm.nd.array(np.zeros(n, dtype=output.dtype), ctx)
    # 2. run tbe kernel
    func(a, b)
    # 3.verify the correctness of output
    result = np.add.reduce(a.asnumpy(), axis=1, keepdims=False)
    try:
        tvm.testing.assert_allclose(b.asnumpy(), result, atol=0.001, rtol=0.001)
    except AssertionError as e:
        print(e)
        return False
    return True


def test_reduce_sum_cpu_api_float16(_):
    """
    for reduce_sum api
    if the dtype is fp16, over than 74, then the results are inconsistent with numpy.
    That is because of numpy use float32 to compute, the result cast to fp16
    @return: Ture && false
    """
    n = 74
    input1 = tvm.placeholder((n, n), name="input1", dtype="float16")
    output = tbe.sum(input1, axis=1)
    sch = tvm.create_schedule(output.op)
    func = tvm.build(sch, [input1, output], "c", "llvm", name="func")
    ctx = tvm.cpu(0)
    # 1. prepare kernel parameter
    a = tvm.nd.array(np.random.uniform(size=(n, n)).astype(input1.dtype), ctx)
    b = tvm.nd.array(np.zeros(n, dtype=output.dtype), ctx)
    # 2. run tbe kernel
    func(a, b)
    # 3.verify the correctness of output
    result = np.add.reduce(a.asnumpy(), axis=1, keepdims=False)
    try:
        tvm.testing.assert_allclose(b.asnumpy(), result, atol=0.001, rtol=0.001)
    except AssertionError as e:
        print(e)
        return False
    return True


def test_reduce_sum_cpu_api_keep_dim(_):
    """
    for reduce_sum api
    @return: Ture && false
    """
    n = 1024
    input1 = tvm.placeholder((n, n), name="input1", dtype="float32")
    output = tbe.sum(input1, axis=0, keepdims=True)
    sch = tvm.create_schedule(output.op)
    func = tvm.build(sch, [input1, output], "c", "llvm", name="func")
    ctx = tvm.cpu(0)
    # 1. prepare kernel parameter
    a = tvm.nd.array(np.random.uniform(size=(n, n)).astype(input1.dtype), ctx)
    b = tvm.nd.array(np.zeros((1, n), dtype=output.dtype), ctx)
    # 2. run tbe kernel
    func(a, b)
    # 3.verify the correctness of output
    result = np.add.reduce(a.asnumpy(), axis=0, keepdims=True)
    try:
        tvm.testing.assert_allclose(b.asnumpy(), result, atol=0.001, rtol=0.001)
    except AssertionError as e:
        print(e)
        return False
    return True


def test_reduce_min_cpu_api(_):
    """
    for reduce_min api
    @return: Ture && false
    """
    n = 1024
    input1 = tvm.placeholder((n, n), name="input1", dtype="float32")
    output = tbe.reduce_min(input1, axis=1)
    sch = tvm.create_schedule(output.op)
    func = tvm.build(sch, [input1, output], "c", "llvm", name="func")
    ctx = tvm.cpu(0)
    # 1. prepare kernel parameter
    a = tvm.nd.array(np.random.uniform(size=(n, n)).astype(input1.dtype), ctx)
    b = tvm.nd.array(np.zeros(n, dtype=output.dtype), ctx)
    # 2. run tbe kernel
    func(a, b)
    # 3.verify the correctness of output
    result = np.minimum.reduce(a.asnumpy(), axis=1, keepdims=False)
    try:
        tvm.testing.assert_allclose(b.asnumpy(), result, atol=0.001, rtol=0.001)
    except AssertionError as e:
        print(e)
        return False
    return True


def test_reduce_min_cpu_api_keep_dim(_):
    """
    for reduce_min api
    @return: Ture && false
    """
    n = 1024
    input1 = tvm.placeholder((n, n), name="input1", dtype="float32")
    output = tbe.reduce_min(input1, axis=0, keepdims=True)
    sch = tvm.create_schedule(output.op)
    func = tvm.build(sch, [input1, output], "c", "llvm", name="func")
    ctx = tvm.cpu(0)
    # 1. prepare kernel parameter
    a = tvm.nd.array(np.random.uniform(size=(n, n)).astype(input1.dtype), ctx)
    b = tvm.nd.array(np.zeros((1, n), dtype=output.dtype), ctx)
    # 2. run tbe kernel
    func(a, b)
    # 3.verify the correctness of output
    result = np.minimum.reduce(a.asnumpy(), axis=0, keepdims=True)
    try:
        tvm.testing.assert_allclose(b.asnumpy(), result, atol=0.001, rtol=0.001)
    except AssertionError as e:
        print(e)
        return False
    return True


def test_reduce_max_cpu_api(_):
    """
    for reduce_max api
    @return: Ture && false
    """
    n = 1024
    input1 = tvm.placeholder((n, n), name="input1", dtype="float32")
    output = tbe.reduce_max(input1, axis=1)
    sch = tvm.create_schedule(output.op)
    func = tvm.build(sch, [input1, output], "c", "llvm", name="func")
    ctx = tvm.cpu(0)
    # 1. prepare kernel parameter
    a = tvm.nd.array(np.random.uniform(size=(n, n)).astype(input1.dtype), ctx)
    b = tvm.nd.array(np.zeros(n, dtype=output.dtype), ctx)
    # 2. run tbe kernel
    func(a, b)
    # 3.verify the correctness of output
    result = np.maximum.reduce(a.asnumpy(), axis=1, keepdims=False)
    try:
        tvm.testing.assert_allclose(b.asnumpy(), result, atol=0.001, rtol=0.001)
    except AssertionError as e:
        print(e)
        return False
    return True


def test_reduce_max_cpu_api_keep_dim(_):
    """
    for reduce_max api
    @return: Ture && false
    """
    n = 1024
    input1 = tvm.placeholder((n, n), name="input1", dtype="float32")
    output = tbe.reduce_max(input1, axis=0, keepdims=True)
    sch = tvm.create_schedule(output.op)
    func = tvm.build(sch, [input1, output], "c", "llvm", name="func")
    ctx = tvm.cpu(0)
    # 1. prepare kernel parameter
    a = tvm.nd.array(np.random.uniform(size=(n, n)).astype(input1.dtype), ctx)
    b = tvm.nd.array(np.zeros((1, n), dtype=output.dtype), ctx)
    # 2. run tbe kernel
    func(a, b)
    # 3.verify the correctness of output
    result = np.maximum.reduce(a.asnumpy(), axis=0, keepdims=True)
    try:
        tvm.testing.assert_allclose(b.asnumpy(), result, atol=0.001, rtol=0.001)
    except AssertionError as e:
        print(e)
        return False
    return True


def test_reduce_prod_cpu_api(_):
    """
    for reduce_prod api
    @return: Ture && false
    """
    n = 1024
    input1 = tvm.placeholder((n, n), name="input1", dtype="float32")
    output = tbe.reduce_prod(input1, axis=1)
    sch = tvm.create_schedule(output.op)
    func = tvm.build(sch, [input1, output], "c", "llvm", name="func")
    ctx = tvm.cpu(0)
    # 1. prepare kernel parameter
    a = tvm.nd.array(np.random.uniform(size=(n, n)).astype(input1.dtype), ctx)
    b = tvm.nd.array(np.zeros(n, dtype=output.dtype), ctx)
    # 2. run tbe kernel
    func(a, b)
    # 3.verify the correctness of output
    result = np.prod(a.asnumpy(), axis=1, keepdims=False)
    try:
        tvm.testing.assert_allclose(b.asnumpy(), result, atol=0.001, rtol=0.001)
    except AssertionError as e:
        print(e)
        return False
    return True


def test_reduce_prod_cpu_api_keep_dim(_):
    """
    for reduce_prod api
    @return: Ture && false
    """
    n = 1024
    input1 = tvm.placeholder((n, n), name="input1", dtype="float32")
    output = tbe.reduce_prod(input1, axis=0, keepdims=True)
    sch = tvm.create_schedule(output.op)
    func = tvm.build(sch, [input1, output], "c", "llvm", name="func")
    ctx = tvm.cpu(0)
    # 1. prepare kernel parameter
    a = tvm.nd.array(np.random.uniform(size=(n, n)).astype(input1.dtype), ctx)
    b = tvm.nd.array(np.zeros((1, n), dtype=output.dtype), ctx)
    # 2. run tbe kernel
    func(a, b)
    # 3.verify the correctness of output
    result = np.prod(a.asnumpy(), axis=0, keepdims=True)
    try:
        tvm.testing.assert_allclose(b.asnumpy(), result, atol=0.001, rtol=0.001)
    except AssertionError as e:
        print(e)
        return False
    return True


test_func_list = [
    test_reduce_sum_cpu_api_float32,
    test_reduce_sum_cpu_api_float16,
    test_reduce_sum_cpu_api_keep_dim,
    test_reduce_min_cpu_api,
    test_reduce_min_cpu_api_keep_dim,
    test_reduce_max_cpu_api,
    test_reduce_max_cpu_api_keep_dim,
    test_reduce_prod_cpu_api,
    test_reduce_prod_cpu_api_keep_dim,
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
