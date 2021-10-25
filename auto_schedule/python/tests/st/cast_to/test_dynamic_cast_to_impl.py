# # -*- coding:utf-8 -*-
from sch_test_frame.ut import OpUT
from sch_test_frame.utils.op_param_util import cartesian_set_format_dtype
from sch_test_frame.common import precision_info
import numpy as np

import tbe
from tbe import tvm
from tbe.common.utils import shape_util
from tbe.common.register import register_operator


@register_operator("cast_to")
def dsl_dynamic_cast_to(x, y, dst_dtype, kernel_name="dsl_dynamic_cast_to"):
    input_dtype = x.get("dtype")

    ins = tbe.dsl.classify([x], "elewise")
    schedules, tensors = [], []

    for (x,) in ins:
        with tbe.dsl.compute():
            shape_x = shape_util.variable_shape([x])[0]
            data1 = tvm.placeholder(shape_x, name='data1', dtype=input_dtype)
            res = tbe.dsl.cast_to(data1, dst_dtype)

            tensors.append((data1, res))

        with tvm.target.cce():
            sch = tbe.dsl.auto_schedule(res)
        schedules.append(sch)

    config = {"name": kernel_name, "tensor_list": tensors}
    tbe.dsl.build(schedules, config)


ut_case = OpUT("cast_to", "cast_to.test_dynamic_cast_to_impl", "dsl_dynamic_cast_to")

case1 = {
    "params": [{
        "shape": (-1, -1),
        "dtype": "float16",
        "range": [(1, None), (1, None)]
    }, {
        "shape": (-1, -1),
        "dtype": "float32",
        "range": [(1, None), (1, None)]
    },
        "float32"],
    "case_name":
        "test_dync_cast_to_1",
    "expect":
        "success",
    "support_expect":
        True
}

case2 = {
    "params": [{
        "shape": (-1, -1),
        "dtype": "float16",
        "range": [(1, None), (1, None)]
    }, {
        "shape": (-1, -1),
        "dtype": "int8",
        "range": [(1, None), (1, None)]
    },
        "int8"],
    "case_name":
        "test_dync_cast_to_2",
    "expect":
        "success",
    "support_expect":
        True
}

case3 = {
    "params": [{
        "shape": (-1, -1),
        "dtype": "float16",
        "range": [(1, None), (1, None)]
    }, {
        "shape": (-1, -1),
        "dtype": "uint8",
        "range": [(1, None), (1, None)]
    },
        "uint8"],
    "case_name":
        "test_dync_cast_to_3",
    "expect":
        "success",
    "support_expect":
        True
}

case4 = {
    "params": [{
        "shape": (-1, -1),
        "dtype": "float16",
        "range": [(1, None), (1, None)]
    }, {
        "shape": (-1, -1),
        "dtype": "int32",
        "range": [(1, None), (1, None)]
    },
        "int32"],
    "case_name":
        "test_dync_cast_to_4",
    "expect":
        "success",
    "support_expect":
        True
}

case5 = {
    "params": [{
        "shape": (-1, -1),
        "dtype": "int8",
        "range": [(1, None), (1, None)]
    }, {
        "shape": (-1, -1),
        "dtype": "float16",
        "range": [(1, None), (1, None)]
    },
        "float16"],
    "case_name":
        "test_dync_cast_to_5",
    "expect":
        "success",
    "support_expect":
        True
}

case6 = {
    "params": [{
        "shape": (-1, -1),
        "dtype": "int8",
        "range": [(1, None), (1, None)]
    }, {
        "shape": (-1, -1),
        "dtype": "uint8",
        "range": [(1, None), (1, None)]
    },
        "uint8"],
    "case_name":
        "test_dync_cast_to_6",
    "expect":
        "success",
    "support_expect":
        True
}

case7 = {
    "params": [{
        "shape": (-1, -1),
        "dtype": "uint8",
        "range": [(1, None), (1, None)]
    }, {
        "shape": (-1, -1),
        "dtype": "float16",
        "range": [(1, None), (1, None)]
    },
        "float16"],
    "case_name":
        "test_dync_cast_to_7",
    "expect":
        "success",
    "support_expect":
        True
}

case8 = {
    "params": [{
        "shape": (-1, -1),
        "dtype": "uint8",
        "range": [(1, None), (1, None)]
    }, {
        "shape": (-1, -1),
        "dtype": "int8",
        "range": [(1, None), (1, None)]
    },
        "int8"],
    "case_name":
        "test_dync_cast_to_8",
    "expect":
        "success",
    "support_expect":
        True
}

case9 = {
    "params": [{
        "shape": (-1, -1),
        "dtype": "int32",
        "range": [(1, None), (1, None)]
    }, {
        "shape": (-1, -1),
        "dtype": "float16",
        "range": [(1, None), (1, None)]
    },
        "float16"],
    "case_name":
        "test_dync_cast_to_9",
    "expect":
        "success",
    "support_expect":
        True
}

case10 = {
    "params": [{
        "shape": (-1, -1),
        "dtype": "int32",
        "range": [(1, None), (1, None)]
    }, {
        "shape": (-1, -1),
        "dtype": "int8",
        "range": [(1, None), (1, None)]
    },
        "int8"],
    "case_name":
        "test_dync_cast_to_10",
    "expect":
        "success",
    "support_expect":
        True
}

case11 = {
    "params": [{
        "shape": (-1, -1),
        "dtype": "int32",
        "range": [(1, None), (1, None)]
    }, {
        "shape": (-1, -1),
        "dtype": "uint8",
        "range": [(1, None), (1, None)]
    },
        "uint8"],
    "case_name":
        "test_dync_cast_to_11",
    "expect":
        "success",
    "support_expect":
        True
}

case12 = {
    "params": [{
        "shape": (-1, -1),
        "dtype": "int32",
        "range": [(1, None), (1, None)]
    }, {
        "shape": (-1, -1),
        "dtype": "float32",
        "range": [(1, None), (1, None)]
    },
        "float32"],
    "case_name":
        "test_dync_cast_to_12",
    "expect":
        "success",
    "support_expect":
        True
}

case13 = {
    "params": [{
        "shape": (-1, -1),
        "dtype": "int32",
        "range": [(1, None), (1, None)]
    }, {
        "shape": (-1, -1),
        "dtype": "int64",
        "range": [(1, None), (1, None)]
    },
        "int64"],
    "case_name":
        "test_dync_cast_to_13_s32_s64",
    "expect":
        "success",
    "support_expect":
        True
}

case14 = {
    "params": [{
        "shape": (-1, -1),
        "dtype": "int64",
        "range": [(1, None), (1, None)]
    }, {
        "shape": (-1, -1),
        "dtype": "int32",
        "range": [(1, None), (1, None)]
    },
        "int32"],
    "case_name":
        "test_dync_cast_to_14_s64_s32",
    "expect":
        "success",
    "support_expect":
        True
}

case15 = {
    "params": [{
        "shape": (-1, -1),
        "dtype": "bfloat16",
        "range": [(1, None), (1, None)]
    }, {
        "shape": (-1, -1),
        "dtype": "float32",
        "range": [(1, None), (1, None)]
    },
        "float32"],
    "case_name":
        "test_dync_cast_to_15_bf16_f32",
    "expect":
        "success",
    "support_expect":
        True
}

ut_case.add_case("all", case1)
ut_case.add_case("all", case2)
ut_case.add_case("all", case3)
ut_case.add_case("all", case4)
ut_case.add_case("all", case5)
ut_case.add_case("all", case6)
ut_case.add_case("all", case7)
ut_case.add_case("all", case8)
ut_case.add_case("all", case9)

ut_case.add_case(["Ascend910A", "Ascend710", "Ascend310"], case10)
ut_case.add_case(["Ascend910A", "Ascend710", "Ascend310"], case11)

ut_case.add_case("all", case12)

# ut_case.add_case(case=case13, support_soc="Ascend920A")
# ut_case.add_case(case=case14, support_soc="Ascend920A")
# ut_case.add_case(case=case15, support_soc="Ascend920A")


def calc_expect_func(x, y, dst_dtype):
    x_value = x.get("value")
    res = x_value.astype(dst_dtype)
    return (res,)


ut_case.add_precision_case(
    "all", {
        "params": [
            {
                "shape": (-1, -1),
                "dtype": "float16",
                "range": [(1, 200), (1, 100)],
                "run_shape": (1, 10),
                "param_type": "input"
            },
            {
                "shape": (-1, -1),
                "dtype": "float32",
                "range": [(1, 200), (1, 100)],
                "run_shape": (1, 10),
                "param_type": "output"
            },
            "float32"
        ],
        "calc_expect_func": calc_expect_func,
        "precision_standard": precision_info.PrecisionStandard(0.0001, 0.0001),
        "case_name": "test_dync_cast_to_prec_01"
    })

ut_case.add_precision_case(
    "all", {
        "params": [
            {
                "shape": (-1, -1),
                "dtype": "float16",
                "range": [(1, 200), (1, 100)],
                "run_shape": (1, 10),
                "param_type": "input"
            },
            {
                "shape": (-1, -1),
                "dtype": "int32",
                "range": [(1, 200), (1, 100)],
                "run_shape": (1, 10),
                "param_type": "output"
            },
            "int32"
        ],
        "calc_expect_func": calc_expect_func,
        "precision_standard": precision_info.PrecisionStandard(0.0001, 0.0001),
        "case_name": "test_dync_cast_to_prec_04"
    })

ut_case.add_precision_case(
    "all", {
        "params": [
            {
                "shape": (-1, -1),
                "dtype": "int8",
                "range": [(1, 200), (1, 100)],
                "run_shape": (1, 10),
                "param_type": "input"
            },
            {
                "shape": (-1, -1),
                "dtype": "float16",
                "range": [(1, 200), (1, 100)],
                "run_shape": (1, 10),
                "param_type": "output"
            },
            "float16"
        ],
        "calc_expect_func": calc_expect_func,
        "precision_standard": precision_info.PrecisionStandard(0.001, 0.001),
        "case_name": "test_dync_cast_to_prec_05"
    })
