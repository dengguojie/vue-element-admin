# # -*- coding:utf-8 -*-
from sch_test_frame.ut import OpUT
import numpy as np
import warnings

from te import tvm
import te.lang.cce as tbe

warnings.filterwarnings("ignore")
ut_case = OpUT("array_cpu", "dsl_cpu.test_array_cpu_impl")


def test_concat_cpu_api_axis_zero(_):
    """
    for concat api
    @return: Ture && false
    """
    n, m = 1024, 5
    input1 = tvm.placeholder((m, n), name="input1", dtype="float16")
    input2 = tvm.placeholder((m, n), name="input2", dtype="float16")
    output = tbe.concat([input1, input2], 0)
    sch = tvm.create_schedule(output.op)
    func = tvm.build(sch, [input1, input2, output], "c", "llvm", name="func")
    ctx = tvm.cpu(0)
    # 1. prepare kernel parameter
    a = tvm.nd.array(np.random.uniform(size=(m, n)).astype(input1.dtype), ctx)
    b = tvm.nd.array(np.random.uniform(size=(m, n)).astype(input1.dtype), ctx)
    c = tvm.nd.array(np.zeros((2 * m, n), dtype=output.dtype), ctx)
    # 2. run tbe kernel
    func(a, b, c)
    # 3.verify the correctness of output
    try:
        tvm.testing.assert_allclose(c.asnumpy(), np.concatenate((a.asnumpy(), b.asnumpy()), axis=0))
    except AssertionError as e:
        print(e)
        return False
    return True


def test_concat_cpu_api_axis_one(_):
    """
    for concat api
    @return: Ture && false
    """
    n, m = 1024, 5
    input1 = tvm.placeholder((m, n), name="input1", dtype="float16")
    input2 = tvm.placeholder((m, n), name="input2", dtype="float16")
    output = tbe.concat([input1, input2], 1)
    sch = tvm.create_schedule(output.op)
    func = tvm.build(sch, [input1, input2, output], "c", "llvm", name="func")
    ctx = tvm.cpu(0)
    # 1. prepare kernel parameter
    a = tvm.nd.array(np.random.uniform(size=(m, n)).astype(input1.dtype), ctx)
    b = tvm.nd.array(np.random.uniform(size=(m, n)).astype(input1.dtype), ctx)
    c = tvm.nd.array(np.zeros((m, 2 * n), dtype=output.dtype), ctx)
    # 2. run tbe kernel
    func(a, b, c)
    # 3.verify the correctness of output
    try:
        tvm.testing.assert_allclose(c.asnumpy(), np.concatenate((a.asnumpy(), b.asnumpy()), axis=1))
    except AssertionError as e:
        print(e)
        return False
    return True


test_func_list = [
    test_concat_cpu_api_axis_zero,
    test_concat_cpu_api_axis_one,
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
