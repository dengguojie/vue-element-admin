# # -*- coding:utf-8 -*-
import numpy as np
import tbe
from sch_test_frame.common import precision_info
from sch_test_frame.ut import OpUT
from tbe import tvm
from tbe.common.register import register_operator
from tbe.common.utils import shape_util
from tbe.dsl import classify


@register_operator("transdata")
def dsl_dync_transdata(x, dst_shape, axes_map, pad_value=0, kernel_name="transdata"):
    with tbe.common.context.op_context.OpContext("dynamic"):
        ins = classify([x, dst_shape, axes_map], "transdata")
        sch_list = []
        tensors = []
        for (x, dst_shape, axes_map) in ins:
            with tbe.dsl.compute():
                src_shape, dst_shape = shape_util.variable_shape([x, dst_shape, axes_map], op_mode="transdata")
                data_input = tvm.placeholder(src_shape, name="data_input", dtype=x.get("dtype"))
                res = tbe.dsl.transdata(data_input, dst_shape, axes_map, pad_value)
                tensor_list = [data_input, res]
                tensors.append(tensor_list)

                with tvm.target.cce():
                    sch = tbe.dsl.auto_schedule(res)
            sch_list.append(sch)
        # build
        config = {"name": kernel_name, "tensor_list": tensors}
        tbe.dsl.build(sch_list, config)


ut_case = OpUT("transdata", "transdata.test_dynamic_transdata_general_impl", "dsl_dync_transdata")

"""
Forward: last+transpose + n-last-transpose
"""
forward_case_0 = {
    "params": [
        {
            "shape": (-1, -1, -1, -1),
            "format": "NCHW",
            "dtype": "float16",
            "range": [(1, None), (1, None), (1, None), (1, None)]
        },
        [-1, -1, -1, -1, 16],
        {0: 0, 1: (1, 4), 2: 2, 3: 3}
    ],
    "case_name":
        "dsl_dync_transdata_0",
    "expect":
        "success",
    "support_expect":
        True
}

forward_case_1 = {
    "params": [
        {
            "shape": (-1, -1, -1, -1),
            "format": "NCHW",
            "dtype": "float32",
            "range": [(1, None), (1, None), (1, None), (1, None)]
        },
        [-1, -1, -1, -1, 16],
        {0: 0, 1: (1, 4), 2: 2, 3: 3}
    ],
    "case_name":
        "dsl_dync_transdata_1",
    "expect":
        "success",
    "support_expect":
        True
}

forward_case_2 = {
    "params": [
        {
            "shape": (-1, -1, -1, -1),
            "format": "NHWC",
            "dtype": "float16",
            "range": [(1, None), (1, None), (1, None), (1, None)]
        },
        [-1, -1, -1, -1, 16],
        {0: 0, 1: 2, 2: 3, 3: (1, 4)}
    ],
    "case_name":
        "dsl_dync_transdata_2",
    "expect":
        "success",
    "support_expect":
        True
}

backward_case_3 = {
    "params": [
        {
            "shape": (-1, -1, -1, -1, 16),
            "format": "NC1HWC0",
            "dtype": "float16",
            "range": [(1, None), (1, None), (1, None), (1, None), (16, 16)]
        },
        [-1, -1, -1, -1],
        {0: 0, 2: 1, 3: 2, (1, 4): 3}
    ],
    "case_name":
        "dsl_dync_transdata_3",
    "expect":
        "success",
    "support_expect":
        True
}

backward_case_4 = {
    "params": [
        {
            "shape": (-1, -1, -1, -1, 16),
            "format": "NC1HWC0",
            "dtype": "float16",
            "range": [(1, None), (1, None), (1, None), (1, None), (16, 16)]
        },
        [-1, -1, -1, -1],
        {0: 0, (1, 4): 1, 2: 2, 3: 3}
    ],
    "case_name":
        "dsl_dync_transdata_4",
    "expect":
        "success",
    "support_expect":
        True
}

backward_case_5 = {
    "params": [
        {
            "shape": (-1, -1, -1, -1, 16),
            "format": "NC1HWC0",
            "dtype": "float32",
            "range": [(1, None), (1, None), (1, None), (1, None), (16, 16)]
        },
        [-1, -1, -1, -1],
        {0: 0, (1, 4): 1, 2: 2, 3: 3}
    ],
    "case_name":
        "dsl_dync_transdata_5",
    "expect":
        "success",
    "support_expect":
        True
}

ut_case.add_case(["Ascend910A", ], forward_case_0)
ut_case.add_case(["Ascend910A", ], forward_case_1)
ut_case.add_case(["Ascend910A", ], forward_case_2)
ut_case.add_case(["Ascend910A", ], backward_case_3)
ut_case.add_case(["Ascend910A", ], backward_case_4)
ut_case.add_case(["Ascend910A", ], backward_case_5)
