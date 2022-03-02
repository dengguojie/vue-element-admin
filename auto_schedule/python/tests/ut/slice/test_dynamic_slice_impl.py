# # -*- coding:utf-8 -*-
import numpy as np
import tbe
from sch_test_frame.common import precision_info
from sch_test_frame.ut import OpUT
from tbe import tvm
from tbe.common.register import register_operator
from tbe.common.utils import shape_util
from tbe.dsl import classify


@register_operator("slice")
def dsl_dynamic_slice(x, begin, size, output, kernel_name="dsl_dynamic_slice"):
    with tbe.common.context.op_context.OpContext("dynamic"):
        ins = classify([x, begin, size], "slice", {"end_mode": "size"})
        schedules, tensors = [], []
        for x_input, begin_input, size_input in ins:
            with tbe.dsl.compute():
                x_shape, begin_list, size_list = shape_util.variable_shape([x_input, begin_input, size_input], "slice")
                x_tensor = tvm.placeholder(x_shape, name="x", dtype=x_input["dtype"])
                begin_tensor = tvm.placeholder([len(begin_list)], name="begin", dtype=begin["dtype"])
                size_tensor = tvm.placeholder([len(size_list)], name="size", dtype=size["dtype"])
                res = tbe.dsl.slice(x_tensor, begin_list, size_list)
                tensors.append([x_tensor, begin_tensor, size_tensor, res])
            with tvm.target.cce():
                sch = tbe.dsl.auto_schedule(res)
            schedules.append(sch)

        # build
        config = {"name": kernel_name, "tensor_list": tensors}
        tbe.dsl.build(schedules, config)

ut_case = OpUT("slice", "slice.test_dynamic_slice_impl", "dsl_dynamic_slice")

case1 = {
    "params": [{
        "shape": (5, -1,),
        "dtype": "float16",
        "range": [(5, 5), (1, None)]
    }, {
        "shape": (2,),
        "dtype": "int32",
        "range": [(2, 2)]
    }, {
        "shape": (2,),
        "dtype": "int32",
        "range": [(2, 2)]
    }, {
        "shape": (5, -1,),
        "dtype": "float16",
        "range": [(5, 5), (1, None)]
    },],
    "case_name":
        "test_dync_slice_1",
    "expect":
        "success",
    "support_expect":
        True
}

case2 = {
    "params": [{
        "shape": (-2,),
        "dtype": "float16",
        "range": [(1, None),]
    }, {
        "shape": (-2,),
        "dtype": "int32",
        "range": [(1, None),]
    }, {
        "shape": (-2,),
        "dtype": "int32",
        "range": [(1, None),]
    },{
        "shape": (-2,),
        "dtype": "float16",
        "range": [(1, None),]
    },],
    "case_name":
        "test_dync_slice_2",
    "expect":
        "success",
    "support_expect":
        True
}

ut_case.add_case(["Ascend910A",], case1)
ut_case.add_case(["Ascend910A",], case2)

if __name__ == '__main__':
    import os
    from pathlib import Path

    _ASCEND_TOOLCHAIN_PATH_ENV = "TOOLCHAIN_HOME"
    simulator_lib_path = Path(os.environ.get(_ASCEND_TOOLCHAIN_PATH_ENV,
                                             "/usr/local/Ascend/toolkit")).joinpath("tools/simulator")
    ut_case.run(["Ascend910A"], simulator_mode="pv", simulator_lib_path=simulator_lib_path)

