# # -*- coding:utf-8 -*-
from sch_test_frame.ut import OpUT
import numpy as np
import math
import operator

from te import tvm
import te.lang.cce as tbe
from tbe.common.testing.testing import debug
from te.platform.cce_conf import te_set_version

ut_case = OpUT("cast_cpu", "dsl_cpu.test_cast_cpu_impl")


def test_cast_cpu_api(soc):
    """
    @param soc: useless parameter for framework
    @return: Ture && false
    """
    cast_dic = {"ceil": [tbe.ceil, np.ceil], "floor": [tbe.floor, np.floor],
                "round": [tbe.round, np.round], "trunc": [tbe.trunc, np.trunc]}
    with debug():
        for api_name, value in cast_dic.items():
            n = 128
            input1 = tvm.placeholder((n, ), name="input1", dtype="float32")
            te_set_version("Ascend910A")
            output = value[0](input1)
            sch = tvm.create_schedule(output.op)
            func = tvm.build(sch, [input1, output], "c", "llvm", name="func")
            ctx = tvm.cpu(0)
            # 1. prepare kernel parameter
            a = tvm.nd.array(np.random.uniform(size=(n, )).astype(input1.dtype), ctx)
            b = tvm.nd.array(np.zeros(n, dtype=output.dtype), ctx)
            # 2. run tbe kernel
            func(a, b)
            # 3.verify the correctness of output
            try:
                # Restore soc version to Ascend310, or it affect the following use cases
                te_set_version("Ascend310")
                tvm.testing.assert_allclose(b.asnumpy(), value[1](a.asnumpy()), rtol=0.001, atol=0.001)
            except AssertionError as e:
                print("\ndsl api name is:", api_name)
                print(e)
                return False
        return True


def test_cast_cpu_api_check_not_support_op_type(soc):
    """
    @param soc: soc version
    @return: Ture && false
    """
    try:
        n = 128
        input1 = tvm.placeholder((n,), name="input1", dtype="float32")
        output = tbe.trunc(input1)
    except RuntimeError as e:
        print("In soc %s," % soc + e.args[0]["detailed_cause"])
    return True


test_func_list = [
    test_cast_cpu_api,
    test_cast_cpu_api_check_not_support_op_type,
]
for item in test_func_list:
    ut_case.add_cust_test_func(test_func=item)


if __name__ == '__main__':
    import os
    from pathlib import Path

    _ASCEND_TOOLCHAIN_PATH_ENV = "TOOLCHAIN_HOME"
    simulator_lib_path = Path(os.environ.get(_ASCEND_TOOLCHAIN_PATH_ENV,
                                             "/usr/local/Ascend/toolkit")).joinpath("tools/simulator")
    ut_case.run(["Ascend310"], simulator_mode="pv", simulator_lib_path=simulator_lib_path)
