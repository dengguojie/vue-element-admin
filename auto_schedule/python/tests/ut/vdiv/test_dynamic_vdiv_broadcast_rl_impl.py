# # -*- coding:utf-8 -*-
import os

from sch_test_frame.ut import OpUT
from sch_test_frame.utils.op_param_util import cartesian_set_format_dtype
from sch_test_frame.common import precision_info
import numpy as np

import tbe
from tbe import tvm
from tbe.common.utils import shape_util
from tbe.common.register import register_operator
from tbe.dsl.base import operation


def dynamic_vdiv_broadcast(x, y, z, kernel_name="dynamic_vdiv_broadcast"):

    input_dtype = x.get("dtype")

    ins = tbe.dsl.classify([x, y], "broadcast")
    schedules, tensors = [], []

    for (x, y) in ins:
        with tbe.dsl.compute():
            shape_x, shape_y = shape_util.variable_shape([x, y])
            shape_x, shape_y = shape_util.refine_shapes_for_broadcast(shape_x, shape_y)
            input_x = tvm.placeholder(shape_x, name='data1', dtype=input_dtype)
            input_y = tvm.placeholder(shape_y, name='data2', dtype=input_dtype)
            x_shape = shape_util.shape_to_list(input_x.shape)
            y_shape = shape_util.shape_to_list(input_y.shape)
            x_shape, y_shape, z_shape = shape_util.broadcast_shapes(x_shape, y_shape,
                                                                    param_name_input1="input_x",
                                                                    param_name_input2="input_y")
            broadcast_x = tbe.dsl.broadcast(input_x, z_shape)
            broadcast_y = tbe.dsl.broadcast(input_y, z_shape)
            res = tbe.dsl.vdiv(broadcast_x, broadcast_y)

            tensors.append((input_x, input_y, res))

        with tvm.target.cce():
            sch = tbe.dsl.auto_schedule(res)
        schedules.append(sch)

    config = {"name": kernel_name, "tensor_list": tensors}
    tbe.dsl.build(schedules, config)

def test_rl_bank_info_for_compile(_):
    tune_bank_path = os.path.dirname(os.path.realpath(__file__))
    os.environ["TUNE_BANK_PATH"] = tune_bank_path

    with tbe.common.context.op_context.OpContext("dynamic"):
        x = {
            "shape": (-1, -1, -1),
            "dtype": "float32",
            "range": [(1, None), (1, None), (1, None)]
        }
        y = {
            "shape": (-1, -1, -1),
            "dtype": "float32",
            "range": [(1, None), (1, None), (1, None)]
        }
        z = {
            "shape": (-1, -1, -1),
            "dtype": "float32",
            "range": [(1, None), (1, None), (1, None)]
        }
        dynamic_vdiv_broadcast(x, y, z, kernel_name="rl_dynamic_vdiv_broadcast")
        compile_info = operation.get_compile_info()
    print(compile_info)
    return "_bank_info" in compile_info.keys()

ut_case = OpUT("vdiv", "vdiv.test_dynamic_vdiv_broadcast_rl_tmpl")
ut_case.add_cust_test_func(support_soc=["Ascend910A"], test_func=test_rl_bank_info_for_compile)


if __name__ == '__main__':
    import os
    from pathlib import Path

    _ASCEND_TOOLCHAIN_PATH_ENV = "TOOLCHAIN_HOME"
    simulator_lib_path = Path(os.environ.get(_ASCEND_TOOLCHAIN_PATH_ENV,
                                             "/usr/local/Ascend/toolkit")).joinpath("tools/simulator")
    ut_case.run(["Ascend910A"], simulator_mode="pv", simulator_lib_path=simulator_lib_path)
