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
def test_reduce_classify(x, x1, axis, keepdims, pre_compile=False, kernel_name="test_reduce_classify"):
    with tbe.common.context.op_context.OpContext("dynamic"):
        x["rel_pos_to_reduce"] = "before"
        x1["rel_pos_to_reduce"] = "after"

        if pre_compile:
            from tbe.common.buildcfg import set_current_build_config
            set_current_build_config("enable_op_prebuild", 1)

        if axis:
            input_axis = {"shape": [len(axis), ], "value": axis, "rel_pos_to_reduce": "axis"}
        else:
            input_axis = {"shape": [-1, ], "rel_pos_to_reduce": "axis", "dtype": "int32"}

        def is_zero():
            shape, range = x.get("shape"), x.get("range")
            for item in zip(shape, range):
                if item[0] == 0 or list(item[1]) == [0, None]:
                    return True
            return False

        if is_zero():
            ins = tbe.dsl.classify([x, input_axis], "reduce", {"keepdims": keepdims})
            for (x, axis) in ins:
                with tbe.dsl.compute():
                    shape_x = shape_util.variable_shape([x, axis], op_mode="reduce")[0]
        else:
            ins = tbe.dsl.classify([x, x1, input_axis], "reduce", {"keepdims": keepdims})
            for (x, x1, axis) in ins:
                with tbe.dsl.compute():
                    shape_x = shape_util.variable_shape([x, x1, axis], op_mode="reduce")[0]


ut_case = OpUT("reduce_sum", "reduce_classify.test_reduce_classify_impl", "test_reduce_classify")

# axis known, range[0, None]
case0 = {
    "params": [{
        "shape": (-1, -1, -1, -1),
        "dtype": "float16",
        "range": [(0, None), (0, None), (0, None), (0, None)]
    }, {
        "shape": (-1, -1),
        "dtype": "float16",
        "range": [(1, None), (1, None)]
    }, [1, 3], False],
    "case_name": "test_reduce_classify_0",
}

# axis known, range[1, None]
case1 = {
    "params": [{
        "shape": (-1, -1, -1, -1),
        "dtype": "float16",
        "range": [(1, None), (1, None), (1, None), (1, None)]
    }, {
        "shape": (-1, -1),
        "dtype": "float16",
        "range": [(1, None), (1, None)]
    }, [0, 2], False],
    "case_name": "test_reduce_classify_1",
}

# axis known, dimX = 0
case2 = {
    "params": [{
        "shape": (-1, -1, 0),
        "dtype": "float16",
        "range": [(1, None), (1, None), (0, 0)]
    }, {
        "shape": (-1, -1),
        "dtype": "float16",
        "range": [(1, None), (1, None)]
    }, [2, ], False],
    "case_name": "test_reduce_classify_2",
}

# axis known, const
case3 = {
    "params": [{
        "shape": (1, 2, 3, 4),
        "dtype": "float16",
        "range": [(1, 1), (2, 2), (3, 3), (4, 4)]
    }, {
        "shape": (1, 3),
        "dtype": "float16",
        "range": [(1, 1), (3, 3)]
    }, [1, 3], False],
    "case_name": "test_reduce_classify_3",
}

# axis known, shape()
case4 = {
    "params": [{
        "shape": (),
        "dtype": "float16",
        "range": []
    }, {
        "shape": (),
        "dtype": "float16",
        "range": [1, 1]
    }, [0], False],
    "case_name": "test_reduce_classify_4",
}

# axis known, pre_compile
case5 = {
    "params": [{
        "shape": (-1, -1, -1, -1),
        "dtype": "float16",
        "range": [(1, None), (1, None), (1, None), (1, None)]
    }, {
        "shape": (-1, -1),
        "dtype": "float16",
        "range": [(1, None), (1, None)]
    }, [0, 2], False, True],
    "case_name": "test_reduce_classify_5",
}

# axis unknown, range[0, None]
caseN0 = {
    "params": [{
        "shape": (-1, -1, -1, -1),
        "dtype": "float16",
        "range": [(0, None), (0, None), (0, None), (0, None)]
    }, {
        "shape": (-1, -1),
        "dtype": "float16",
        "range": [(1, None), (1, None)]
    }, None, False],
    "case_name": "test_reduce_classify_unknown_0",
}

# axis unknown, range[1, None]
caseN1 = {
    "params": [{
        "shape": (-1, -1, -1, -1),
        "dtype": "float16",
        "range": [(1, None), (1, None), (1, None), (1, None)]
    }, {
        "shape": (-1, -1),
        "dtype": "float16",
        "range": [(1, None), (1, None)]
    }, None, False],
    "case_name": "test_reduce_classify_unknown_1",
}

# axis unknown, dimX = 0
caseN2 = {
    "params": [{
        "shape": (-1, -1, 0),
        "dtype": "float16",
        "range": [(1, None), (1, None), (0, 0)]
    }, {
        "shape": (-1, -1),
        "dtype": "float16",
        "range": [(1, None), (1, None)]
    }, None, False],
    "case_name": "test_reduce_classify_unknown_2",
}

# axis unknown, const
caseN3 = {
    "params": [{
        "shape": (1, 2, 3, 4),
        "dtype": "float16",
        "range": [(1, 1), (2, 2), (3, 3), (4, 4)]
    }, {
        "shape": (1, 3),
        "dtype": "float16",
        "range": [(1, 1), (3, 3)]
    }, None, False],
    "case_name": "test_reduce_classify_unknown_3",
}

# axis unknown, shape()
caseN4 = {
    "params": [{
        "shape": (),
        "dtype": "float16",
        "range": []
    }, {
        "shape": (),
        "dtype": "float16",
        "range": [1, 1]
    }, None, False],
    "case_name": "test_reduce_classify_unknown_4",
}

# axis known, pre_compile
caseN5 = {
    "params": [{
        "shape": (-1, -1, -1, -1),
        "dtype": "float16",
        "range": [(1, None), (1, None), (1, None), (1, None)]
    }, {
        "shape": (-1, -1),
        "dtype": "float16",
        "range": [(1, None), (1, None)]
    }, None, False, True],
    "case_name": "test_reduce_classify_unknown_5",
}

# ERROR CASE
caseE0 = {
    "params": [{
        "shape": (-1, -2, -1, -1),
        "dtype": "float16",
        "range": [(1, None), (1, None), (1, None), (1, None)]
    }, {
        "shape": (-1, -1),
        "dtype": "float16",
        "range": [(1, None), (1, None)]
    }, [1, 3], False],
    "case_name": "test_reduce_classify_unknown_5",
    "expect": "failed"
}

caseE1 = {
    "params": [{
        "shape": (-1, -1, -1, -1),
        "dtype": "float16",
        "range": [(1, None), (1, None), (1, None), (1, None)]
    }, {
        "shape": (-1, -1),
        "dtype": "float16",
        "range": [(1, None), (1, None)]
    }, None, 1, False],
    "case_name": "test_reduce_classify_unknown_5",
    "expect": "failed"
}

ut_case.add_case(["Ascend310", "Ascend910A"], case0)
ut_case.add_case(["Ascend310", "Ascend910A"], case1)
ut_case.add_case(["Ascend310", "Ascend910A"], case2)
ut_case.add_case(["Ascend310", "Ascend910A"], case3)
ut_case.add_case(["Ascend310", "Ascend910A"], case4)

ut_case.add_case(["Ascend310", "Ascend910A"], caseN0)
ut_case.add_case(["Ascend310", "Ascend910A"], caseN1)
ut_case.add_case(["Ascend310", "Ascend910A"], caseN2)
ut_case.add_case(["Ascend310", "Ascend910A"], caseN3)
ut_case.add_case(["Ascend310", "Ascend910A"], caseN4)

ut_case.add_case(["Ascend310", "Ascend910A"], caseE0)
ut_case.add_case(["Ascend310", "Ascend910A"], caseE1)
ut_case.add_case(["Ascend310", "Ascend910A"], case5)
ut_case.add_case(["Ascend310", "Ascend910A"], caseN5)
