# # -*- coding:utf-8 -*-
from sch_test_frame.ut import OpUT
import numpy as np
import operator
import warnings

from te import tvm
import te.lang.cce as tbe
from tbe.common.testing.testing import debug

warnings.filterwarnings("ignore")
ut_case = OpUT("inplace_cpu", "dsl_cpu.test_inplace_cpu_impl")


def _get_benchmark_data(op_func_name, op_func, a, b):
    if op_func_name != "inplace_update":
        return np.array(
                   [
                    a.asnumpy()[0],
                    op_func(op_func(a.asnumpy()[1], b.asnumpy()[0]), b.asnumpy()[1]),
                    op_func(op_func(a.asnumpy()[2], b.asnumpy()[3]), b.asnumpy()[4]),
                    a.asnumpy()[3],
                    op_func(a.asnumpy()[4], b.asnumpy()[2]),
                    a.asnumpy()[5]
                   ]
               )
    else:
        return np.array(
                    [
                        a.asnumpy()[0],
                        b.asnumpy()[1],
                        b.asnumpy()[4],
                        a.asnumpy()[3],
                        b.asnumpy()[2],
                        a.asnumpy()[5]
                    ]
                )


def test_inplace_cpu_api(_):
    """
    @return: Ture && false
    """
    inplace_dic = {"inplace_add": [tbe.inplace_add, operator.add], "inplace_sub": [tbe.inplace_sub, operator.sub],
                   "inplace_update": [tbe.inplace_update, None]}
    with debug():
        for api_name, value in inplace_dic.items():
            n = 1024
            input1 = tvm.placeholder((6, n), name="input1", dtype="float32")
            input2 = tvm.placeholder((5, n), name="input2", dtype="float32")
            # index in B:  0, 1, 2, 3, 4
            inplace_ids = [1, 1, 4, 2, 2]
            output = value[0](input1, inplace_ids, input2)
            sch = tvm.create_schedule(output.op)
            func = tvm.build(sch, [input1, input2, output], "c", "llvm", name="func")
            ctx = tvm.cpu(0)
            # 1. prepare kernel parameter
            a = tvm.nd.array(np.random.uniform(size=(6, n)).astype(input1.dtype), ctx)
            b = tvm.nd.array(np.random.uniform(size=(5, n)).astype(input1.dtype), ctx)
            c = tvm.nd.array(np.zeros((6, n), dtype=output.dtype), ctx)
            # 2. run tbe kernel
            func(a, b, c)
            # 3.verify the correctness of output
            try:
                tvm.testing.assert_allclose(c.asnumpy(), _get_benchmark_data(api_name, value[1], a, b))
            except AssertionError as e:
                print("\ndsl api name is:", api_name)
                print(e)
                return False
        return True


test_func_list = [
    test_inplace_cpu_api,
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
