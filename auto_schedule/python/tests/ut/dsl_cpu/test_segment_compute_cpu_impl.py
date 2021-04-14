# # -*- coding:utf-8 -*-
from sch_test_frame.ut import OpUT
import numpy as np
import warnings

from te import tvm
import te.lang.cce as tbe

warnings.filterwarnings("ignore")
ut_case = OpUT("segment_cpu", "dsl_cpu.test_segment_cpu_impl")


def _mean_(value1, value2):
    return (value2 + value1) / 2


def test_segment_cpu_api(_):
    """
    for segment api
    @return: Ture && false
    """
    segment_op = {"segment_max":  [tbe.unsorted_segment_max, np.maximum],
                  "segment_min":  [tbe.unsorted_segment_min, np.minimum],
                  "segment_prod": [tbe.unsorted_segment_prod, np.multiply],
                  "segment_sum":  [tbe.unsorted_segment_sum, np.add],
                  "segment_mean": [tbe.unsorted_segment_mean, _mean_],
                  }
    for _, value in segment_op.items():
        n, m = 1024, 5
        input1 = tvm.placeholder((m, n), name="input1", dtype="float32")
        # index:       0, 1, 2, 3, 4
        segment_ids = [1, 1, 4, 5, 5]
        num_segments = 6
        output = value[0](input1, segment_ids, num_segments)
        sch = tvm.create_schedule(output.op)
        func = tvm.build(sch, [input1, output], "c", "llvm", name="func")
        ctx = tvm.cpu(0)
        # 1. prepare kernel parameter
        a = tvm.nd.array(np.random.uniform(size=(m, n)).astype(input1.dtype), ctx)
        b = tvm.nd.array(np.zeros((m + 1, n), dtype=output.dtype), ctx)
        # 2. run tbe kernel
        func(a, b)
        # 3.verify the correctness of output
        result = np.array([
            np.zeros(n),
            value[1](a.asnumpy()[0], a.asnumpy()[1]),
            np.zeros(n),
            np.zeros(n),
            a.asnumpy()[2],
            value[1](a.asnumpy()[3], a.asnumpy()[4])
        ])
        try:
            tvm.testing.assert_allclose(b.asnumpy(), result)
        except AssertionError as e:
            print(e)
            return False
    return True


def test_segment_cpu_api_segment_id_is_tensor(_):
    """
    for segment api
    only max, min, prod support segment_id is a tensor
    @return: Ture && false
    """
    segment_op = {
        "segment_max":  [tbe.unsorted_segment_max, np.maximum, 0],
        "segment_min":  [tbe.unsorted_segment_min, np.minimum, 0],
        "segment_prod": [tbe.unsorted_segment_prod, np.multiply, 1],
    }
    for op_name, value in segment_op.items():
        n, m = 1000, 5
        input1 = tvm.placeholder((m, n), name="input1", dtype="float32")
        input2 = tvm.placeholder((m, ), name="input2", dtype="int32")
        num_segments = 6
        init_value = value[2]
        output = value[0](input1, input2, num_segments, init_value)
        sch = tvm.create_schedule(output.op)
        func = tvm.build(sch, [input1, input2, output], "c", "llvm", name="func")
        ctx = tvm.cpu(0)
        # 1. prepare kernel parameter
        a = tvm.nd.array(np.random.uniform(size=(m, n)).astype(input1.dtype), ctx)
        b = tvm.nd.array(np.array([1, 1, 4, 5, 5]).astype(input2.dtype), ctx)
        c = tvm.nd.array(np.zeros((m + 1, n), dtype=output.dtype), ctx)
        # 2. run tbe kernel
        func(a, b, c)
        # 3.verify the correctness of output
        result = np.array([
            np.full((n, ), init_value),
            value[1](value[1](init_value, a.asnumpy()[0]), a.asnumpy()[1]),
            np.full((n, ), init_value),
            np.full((n, ), init_value),
            value[1](init_value, a.asnumpy()[2]),
            value[1](value[1](init_value, a.asnumpy()[3]), a.asnumpy()[4])
        ])
        try:
            tvm.testing.assert_allclose(c.asnumpy(), result)
        except AssertionError as e:
            print("\nsegment_op is ", op_name)
            print(e)
            return False
    return True


def test_segment_cpu_api_segment_id_is_minus(_):
    """
    for segment api
    max(segment_id) < 0
    @return: Ture && false
    """
    segment_op = {"segment_max":  [tbe.unsorted_segment_max, np.maximum, 1],
                  "segment_min":  [tbe.unsorted_segment_min, np.minimum, 1],
                  "segment_prod": [tbe.unsorted_segment_prod, np.multiply, 1],
                  "segment_sum":  [tbe.unsorted_segment_sum, np.add, 1],
                  "segment_mean": [tbe.unsorted_segment_mean, _mean_, 1],
                  }
    for _, value in segment_op.items():
        n, m = 1000, 5
        input1 = tvm.placeholder((m, n), name="input1", dtype="float32")
        # index:        0,  1,  2,  3,  4
        segment_ids = [-1, -1, -4, -5, -5]
        num_segments = 6
        init_value = value[2]
        output = value[0](input1, segment_ids, num_segments, init_value)
        sch = tvm.create_schedule(output.op)
        func = tvm.build(sch, [input1, output], "c", "llvm", name="func")
        ctx = tvm.cpu(0)
        # 1. prepare kernel parameter
        a = tvm.nd.array(np.random.uniform(size=(m, n)).astype(input1.dtype), ctx)
        b = tvm.nd.array(np.zeros((m + 1, n), dtype=output.dtype), ctx)
        # 2. run tbe kernel
        func(a, b)
        # 3.verify the correctness of output
        result = np.array([
            np.full((n, ), init_value),
            np.full((n, ), init_value),
            np.full((n, ), init_value),
            np.full((n, ), init_value),
            np.full((n, ), init_value),
            np.full((n, ), init_value)
        ])
        try:
            tvm.testing.assert_allclose(b.asnumpy(), result)
        except AssertionError as e:
            print(e)
            return False
    return True


test_func_list = [
    test_segment_cpu_api,
    test_segment_cpu_api_segment_id_is_tensor,
    test_segment_cpu_api_segment_id_is_minus,
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
