# # -*- coding:utf-8 -*-
from sch_test_frame.ut import OpUT
from sch_test_frame.utils.op_param_util import cartesian_set_format_dtype
from sch_test_frame.common import precision_info
import numpy as np

import tbe
from tbe import tvm
from tbe.common.utils import shape_util
from tbe.common.register import register_operator


@register_operator("reduce_sum")
def test_dynamic_reduce_pattern_parser(_):
    with tbe.common.context.op_context.OpContext("dynamic"):
    
        x = {"shape": (32, 8, 256, 64, 16), "dtype": "float32", "format": "ND"}
        axis = [0, 2]
        keepdims = False
        input_dtype = x.get("dtype")
        x["rel_pos_to_reduce"] = 'before'
        input_axis = {"shape": [len(axis), ], "value": axis, "rel_pos_to_reduce": "axis"}
        ins = tbe.dsl.classify([x, input_axis], "reduce", {"keepdims": keepdims is True})

        schedules, tensors = [], []

        try:
            for (x, axis) in ins:
                with tbe.dsl.compute():
                    shape_x = shape_util.variable_shape([x, axis], op_mode="reduce")[0]
                    data1 = tvm.placeholder(shape_x, name='data1', dtype=input_dtype)
                    sum1 = tbe.dsl.reduce_sum(data1, axis.get("value"), keepdims)
                    res = tbe.dsl.reduce_sum(sum1, 1, keepdims)
                    tensors.append([data1, res])

                with tvm.target.cce():
                    sch = tbe.dsl.auto_schedule(res)
                schedules.append(sch)
                
                
                config = {"name": kernel_name, "tensor_list": tensors}
                tbe.dsl.build(schedules, config)
        except Exception as e:
          
            return True
        return False


ut_case = OpUT("reduce_patternparser", "reduce_patternparser.test_dynamic_reduce_pattern_parser_impl", "test_dynamic_reduce_pattern_parser")



compile_case_list = [
    test_dynamic_reduce_pattern_parser,
]
for item in compile_case_list:
    ut_case.add_cust_test_func(test_func=item)


if __name__ == '__main__':
    import os
    from pathlib import Path

    _ASCEND_TOOLCHAIN_PATH_ENV = "TOOLCHAIN_HOME"
    simulator_lib_path = Path(os.environ.get(_ASCEND_TOOLCHAIN_PATH_ENV,
                                             "/usr/local/Ascend/toolkit")).joinpath("tools/simulator")
    ut_case.run(["Ascend910A"], simulator_mode="pv", simulator_lib_path=simulator_lib_path)
