# # -*- coding:utf-8 -*-
import numpy as np
import tbe
from sch_test_frame.common import precision_info
from sch_test_frame.ut import OpUT
from tbe import tvm
from tbe.common.register import register_operator
from tbe.common.utils import shape_util
from tbe.dsl import classify
from tbe.common.context import op_info


@register_operator("UT_Transpose")
def ut_transpose(x, y, perm, kernel_name="ut_transpose"):
    with tbe.common.context.op_context.OpContext("static") as f:
        opInfo = op_info.OpInfo("Transpose", "Transpose")
        f.add_op_info(opInfo)
        dtype_x = x.get("dtype")

        extra_params = {"axes": perm}
        ins = classify([x], "transpose", extra_params)
        schedules, tensors = [], []
        for (input_x_, perm_) in ins:
            with tbe.dsl.compute():
                shape_x = shape_util.variable_shape([input_x_], "transpose")
                data_x = tvm.placeholder(shape_x, dtype=dtype_x, name="data_x")
                res = tbe.dsl.transpose(data_x, perm_)

                tensors.append([data_x, res])
            with tvm.target.cce():
                sch = tbe.dsl.auto_schedule(res)
            schedules.append(sch)

        config = {"name": kernel_name, "tensor_list": tensors}
        tbe.dsl.build(schedules, config)
        compile_info = tbe.dsl.base.operation.get_compile_info()
        import json
        print(json.dumps(compile_info))


ut_case = OpUT("UT_Transpose", "transpose.test_static_transpose_impl", "ut_transpose")

rl_bank_case0 = {
    "params": [
        {"shape": [444, 32, 200], "dtype": "uint8", "range": [[444, 444], [32, 32], [200, 200]], 'format': 'ND'}, 
        {"shape": [200, 32, 444], "dtype": "uint8", "range": [[200, 200], [32, 32], [444, 444]], 'format': 'ND'}, 
        [2, 1, 0]],
    "case_name": "test_static_rl_bank_transpose_0",
    "expect": "success",
    "support_expect": True
}

# rl_bank_case0 for static_Ascend910A rl bank
ut_case.add_case(["Ascend910A"], rl_bank_case0)


if __name__ == '__main__':
    import os
    from pathlib import Path

    _ASCEND_TOOLCHAIN_PATH_ENV = "TOOLCHAIN_HOME"
    simulator_lib_path = Path(os.environ.get(_ASCEND_TOOLCHAIN_PATH_ENV,
                                             "/usr/local/Ascend/toolkit")).joinpath("tools/simulator")
    ut_case.run(["Ascend910A"], simulator_mode="pv", simulator_lib_path=simulator_lib_path)
