# # -*- coding:utf-8 -*-
from sch_test_frame.ut import OpUT
import tbe
from tbe.common.utils import shape_util

BEFORE = "before"
AFTER = "after"
AXIS = "axis"


def test_reduce_classify(inputs, axis=None, keepdims=None, pre_compile=False, kernel_name="test_reduce_classify"):
    """
    inputs: [{"shape": (-1), "dtype": "float16", "range": [(0, None),], "rel_pos_to_reduce": BEFORE},
             {"shape": (-1), "dtype": "float16", "range": [(0, None),], "rel_pos_to_reduce": BEFORE},]
    axis: int, [], [A,B,C,...], None
    """
    with tbe.common.context.op_context.OpContext("dynamic"):
        if pre_compile:
            from tbe.common.buildcfg import set_current_build_config
            set_current_build_config("enable_op_prebuild", 1)

        if axis:
            input_axis = {"shape": [len(axis), ], "value": axis, "rel_pos_to_reduce": AXIS}
        elif axis == []:
            # -2 can't assure axis's length
            input_axis = {"shape": [-1, ], "rel_pos_to_reduce": AXIS, "value": [], "dtype": "int32"}
        else:
            input_axis = {"shape": [-1, ], "rel_pos_to_reduce": AXIS, "dtype": "int32"}

        def is_zero():
            for _x in inputs:
                if _x.get("rel_pos_to_reduce") != BEFORE:
                    continue
                _shape, _range = _x.get("shape"), _x.get("range")
                for item in zip(_shape, _range):
                    if item[0] == 0 or list(item[1]) == [0, None]:
                        return True
                return False

        inputs.append(input_axis)
        if is_zero():
            ins = tbe.dsl.classify(inputs, "reduce", {"keepdims": keepdims})
            for item in ins:
                with tbe.dsl.compute():
                    shape_x = shape_util.variable_shape(list(item), op_mode="reduce")[0]
        else:
            ins = tbe.dsl.classify(inputs, "reduce", {"keepdims": keepdims})
            for item in ins:
                with tbe.dsl.compute():
                    shape_x = shape_util.variable_shape(list(item), op_mode="reduce")[0]


ut_case = OpUT("reduce_sum", "reduce_classify.test_reduce_classify_impl", "test_reduce_classify")

# axis known, range[0, None]
case0 = {
    "params": [
        [{"shape": (-1, -1, -1, -1), "dtype": "float16",
          "range": [(0, None), (0, None), (0, None), (0, None)],
          "rel_pos_to_reduce": BEFORE},
         {"shape": (-1, -1), "dtype": "float16",
          "range": [(1, None), (1, None)],
          "rel_pos_to_reduce": AFTER}, ],
        [1, 3], False, False
    ],
    "case_name": "test_reduce_classify_0",
}

# axis known, range[1, None]
case1 = {
    "params": [
        [{"shape": (-1, -1, -1, -1), "dtype": "float16",
          "range": [(1, None), (1, None), (1, None), (1, None)],
          "rel_pos_to_reduce": BEFORE},
         {"shape": (-1, -1), "dtype": "float16",
          "range": [(1, None), (1, None)],
          "rel_pos_to_reduce": AFTER}, ],
        [0, 2], False, False
    ],
    "case_name": "test_reduce_classify_1",
}

# axis known, dimX = 0
case2 = {
    "params": [
        [{"shape": (-1, -1, 0), "dtype": "float16",
          "range": [(1, None), (1, None), (0, 0)],
          "rel_pos_to_reduce": BEFORE},
         {"shape": (-1, -1), "dtype": "float16",
          "range": [(1, None), (1, None)],
          "rel_pos_to_reduce": AFTER}, ],
        [2, ], False, False
    ],
    "case_name": "test_reduce_classify_2",
}

# axis known, const
case3 = {
    "params": [
        [{"shape": (1, 2, 3, 4), "dtype": "float16",
          "range": [(1, 1), (2, 2), (3, 3), (4, 4)],
          "rel_pos_to_reduce": BEFORE},
         {"shape": (1, 3), "dtype": "float16",
          "range": [(1, 1), (3, 3)],
          "rel_pos_to_reduce": AFTER}, ],
        [1, 3], False, False
    ],
    "case_name": "test_reduce_classify_3",
}

# axis known, shape()
case4 = {
    "params": [
        [{"shape": (), "dtype": "float16",
          "range": [],
          "rel_pos_to_reduce": BEFORE},
         {"shape": (), "dtype": "float16",
          "range": [1, 1],
          "rel_pos_to_reduce": AFTER}, ],
        [0, ], False, False
    ],
    "case_name": "test_reduce_classify_4",
}

# axis unknown, range[0, None]
case5 = {
    "params": [
        [{"shape": (-1, -1, -1, -1), "dtype": "float16",
          "range": [(0, None), (0, None), (0, None), (0, None)],
          "rel_pos_to_reduce": BEFORE},
         {"shape": (-1, -1), "dtype": "float16",
          "range": [(1, None), (1, None)],
          "rel_pos_to_reduce": AFTER}, ],
        None, False, False
    ],
    "case_name": "test_reduce_classify_5",
}

# axis unknown, range[1, None]
case6 = {
    "params": [
        [{"shape": (-1, -1, -1, -1), "dtype": "float16",
          "range": [(1, None), (1, None), (1, None), (1, None)],
          "rel_pos_to_reduce": BEFORE},
         {"shape": (-1, -1), "dtype": "float16",
          "range": [(1, None), (1, None)],
          "rel_pos_to_reduce": AFTER}, ],
        None, False, False
    ],
    "case_name": "test_reduce_classify_6",
}

# axis unknown, dimX = 0
case7 = {
    "params": [
        [{"shape": (-1, -1, 0), "dtype": "float16",
          "range": [(1, None), (1, None), (0, 0)],
          "rel_pos_to_reduce": BEFORE},
         {"shape": (-1, -1), "dtype": "float16",
          "range": [(1, None), (1, None)],
          "rel_pos_to_reduce": AFTER}, ],
        None, False, False
    ],
    "case_name": "test_reduce_classify_7",
}

# axis unknown, const
case8 = {
    "params": [
        [{"shape": (1, 2, 3, 4), "dtype": "float16",
          "range": [(1, 1), (2, 2), (3, 3), (4, 4)],
          "rel_pos_to_reduce": BEFORE},
         {"shape": (1, 3), "dtype": "float16",
          "range": [(1, 1), (3, 3)],
          "rel_pos_to_reduce": AFTER}, ],
        None, False, False
    ],
    "case_name": "test_reduce_classify_8",
}

# axis unknown, shape()
case9 = {
    "params": [
        [{"shape": (), "dtype": "float16",
          "range": [],
          "rel_pos_to_reduce": BEFORE},
         {"shape": (), "dtype": "float16",
          "range": [],
          "rel_pos_to_reduce": AFTER}, ],
        None, False, False
    ],
    "case_name": "test_reduce_classify_9",
}

"""
CASE: -2
"""
# axis known(int), keep_dims=False, -1 and -2 mixed, before_infer_after
case10 = {
    "params": [
        [{"shape": (-1, -1, -1, -1), "dtype": "float16",
          "range": [(1, None), (1, None), (1, None), (1, None)],
          "rel_pos_to_reduce": BEFORE},
         {"shape": (-2,), "dtype": "float16",
          "range": [],
          "rel_pos_to_reduce": AFTER}, ],
        0, False, False
    ],
    "case_name": "test_reduce_classify_10",
}

# axis known(list), keep_dims=False, -1 and -2 mixed, after_infer_before
case11 = {
    "params": [
        [{"shape": (-1, -1, -1, -1), "dtype": "float16",
          "range": [(1, None), (1, None), (1, None), (1, None)],
          "rel_pos_to_reduce": AFTER},
         {"shape": (-2,), "dtype": "float16",
          "range": [],
          "rel_pos_to_reduce": BEFORE}, ],
        [1, 2], False, False
    ],
    "case_name": "test_reduce_classify_11",
}

# axis known([]), keep_dims=False, all -2
case12 = {
    "params": [
        [{"shape": (-2,), "dtype": "float16",
          "range": [],
          "rel_pos_to_reduce": AFTER},
         {"shape": (-2,), "dtype": "float16",
          "range": [],
          "rel_pos_to_reduce": BEFORE}, ],
        [], False, False
    ],
    "case_name": "test_reduce_classify_12",
}

# axis unkown, keep_dims=False, all -2
case13 = {
    "params": [
        [{"shape": (-2,), "dtype": "float16",
          "range": [],
          "rel_pos_to_reduce": AFTER},
         {"shape": (-2,), "dtype": "float16",
          "range": [],
          "rel_pos_to_reduce": BEFORE}, ],
        None, False, False
    ],
    "case_name": "test_reduce_classify_13",
}

# axis known(list), keep_dims=False, -1 and -2 mixed, after_infer_before
case14 = {
    "params": [
        [{"shape": (-1, -1, -1, -1), "dtype": "float16",
          "range": [(1, None), (1, None), (1, None), (1, None)],
          "rel_pos_to_reduce": AFTER},
         {"shape": (-2,), "dtype": "float16",
          "range": [],
          "rel_pos_to_reduce": BEFORE}, ],
        [1, 2], True, False
    ],
    "case_name": "test_reduce_classify_14",
}

# axis known(list), keep_dims=False, -2 shape, after_infer_before,reduce axis value is negative
case15 = {
    "params": [
        [{"shape": (-2,), "dtype": "float16",
          "range": [(1, None)],
          "rel_pos_to_reduce": AFTER},
         {"shape": (-2,), "dtype": "float16",
          "range": [(1, None)],
          "rel_pos_to_reduce": BEFORE}, ],
        [-1, ], True, False
    ],
    "case_name": "test_reduce_classify_14",
}

"""
CASE: ERROR
"""
caseE0 = {
    "params": [
        [{"shape": (-1, -1, -1, -1), "dtype": "float16",
          "range": [(1, None), (1, None), (1, None), (1, None)],
          "rel_pos_to_reduce": BEFORE},
         {"shape": (-1, -1), "dtype": "float16",
          "range": [(1, None), (1, None)],
          "rel_pos_to_reduce": AFTER}, ],
        None, 1, False
    ],
    "case_name": "test_reduce_classify_E0",
    "expect": "failed"
}

"""
CASE: pre_compile
"""
# axis unknown
caseP0 = {
    "params": [
        [{"shape": (-1, -1, -1, -1), "dtype": "float16",
          "range": [(1, None), (1, None), (1, None), (1, None)],
          "rel_pos_to_reduce": BEFORE},
         {"shape": (-1, -1), "dtype": "float16",
          "range": [(1, None), (1, None)],
          "rel_pos_to_reduce": AFTER}, ],
        None, False, True
    ],
    "case_name": "test_reduce_classify_P0",
}

# axis known
caseP1 = {
    "params": [
        [{"shape": (-1, -1, -1, -1), "dtype": "float16",
          "range": [(1, None), (1, None), (1, None), (1, None)],
          "rel_pos_to_reduce": BEFORE},
         {"shape": (-1, -1), "dtype": "float16",
          "range": [(1, None), (1, None)],
          "rel_pos_to_reduce": AFTER}, ],
        [0, 2], False, True
    ],
    "case_name": "test_reduce_classify_P1",
}

# -1 -2 mixed
caseP2 = {
    "params": [
        [{"shape": (-1, -1, -1, -1), "dtype": "float16",
          "range": [(1, None), (1, None), (1, None), (1, None)],
          "rel_pos_to_reduce": AFTER},
         {"shape": (-2,), "dtype": "float16",
          "range": [],
          "rel_pos_to_reduce": BEFORE}, ],
        [1, 2], True, True
    ],
    "case_name": "test_reduce_classify_P2",
}

ut_case.add_case(["Ascend910A"], case0)
ut_case.add_case(["Ascend910A"], case1)
ut_case.add_case(["Ascend910A"], case2)
ut_case.add_case(["Ascend910A"], case3)
ut_case.add_case(["Ascend910A"], case4)
ut_case.add_case(["Ascend910A"], case5)
ut_case.add_case(["Ascend910A"], case6)
ut_case.add_case(["Ascend910A"], case7)
ut_case.add_case(["Ascend910A"], case8)
ut_case.add_case(["Ascend910A"], case9)
ut_case.add_case(["Ascend910A"], case10)
ut_case.add_case(["Ascend910A"], case11)
ut_case.add_case(["Ascend910A"], case12)
ut_case.add_case(["Ascend910A"], case13)
ut_case.add_case(["Ascend910A"], case14)
ut_case.add_case(["Ascend910A"], case15)

ut_case.add_case(["Ascend910A"], caseE0)

ut_case.add_case(["Ascend910A"], caseP0)
ut_case.add_case(["Ascend910A"], caseP1)
ut_case.add_case(["Ascend910A"], caseP2)
