# # -*- coding:utf-8 -*-
import tbe
from sch_test_frame.ut import OpUT
from tbe import tvm
from tbe.common.register import register_operator
from tbe.common.utils import shape_util
from tbe.common import context as tbe_context
from tbe.dsl import classify
from tbe.common.utils import para_check


@register_operator("para_check")
@para_check.check_op_params(para_check.REQUIRED_INPUT, para_check.OPTION_INPUT, para_check.REQUIRED_OUTPUT,
                            para_check.OPTION_OUTPUT, para_check.KERNEL_NAME)
def dsl_dync_para_check(x, y, z, k, kernel_name="dsl_para_check"):
    """

    Parameters
    ----------
    Algorithm: para_check

    Parameters:

    x: the dict of input data, support float16

    y: the dict of output

    kernel_name: cce kernel name, default value is "para_check".

    Returns
    -------
    None
    """

    input_dtype = x.get("dtype")

    ins = classify([x, y], "elewise")
    schedules, tensors = [], []

    for (x, y) in ins:
        with tbe.dsl.compute():
            shape_x, shape_y = shape_util.variable_shape([x, y])
            data1 = tvm.placeholder(shape_x, name='data1', dtype=input_dtype)
            data2 = tvm.placeholder(shape_y, name='data2', dtype=input_dtype)
            res = tbe.dsl.vadd(data1, data2)

            tensors.append((data1, data2, res))

        with tvm.target.cce():
            sch = tbe.dsl.auto_schedule(res)
        schedules.append(sch)

    config = {"name": kernel_name, "tensor_list": tensors}
    tbe.dsl.build(schedules, config)

    tbe_context.get_context().add_compile_info("ops_compile_info", "ops_compile_info")


ut_case = OpUT("para_check", "para_check.test_dynamic_para_check_impl", "dsl_dync_para_check")

case1 = {
    "params": [
        {
            "shape": (-1, 16),
            "ori_shape": (-1, 16),
            "dtype": "float16",
            "range": [(1, None), (16, 16)],
            "format": "ND",
            "ori_format": "ND",
        },
        {
            "shape": (-1, 16),
            "ori_shape": (-1, 16),
            "dtype": "float16",
            "range": [(1, None), (16, 16)],
            "format": "ND",
            "ori_format": "ND",
        },
        {
            "shape": (-1, 16),
            "ori_shape": (-1, 16),
            "dtype": "float16",
            "range": [(1, None), (16, 16)],
            "format": "ND",
            "ori_format": "ND",
        },
        {
            "shape": (-1, 16),
            "ori_shape": (-1, 16),
            "dtype": "float16",
            "range": [(1, None), (16, 16)],
            "format": "ND",
            "ori_format": "ND",
        }],
    "case_name":
        "test_dynamic_para_check_impl_1",
    "expect":
        "success",
    "support_expect":
        True
}

# SHAPE
case2 = {
    "params": [
        {
            "ori_shape": (-1, 16),
            "dtype": "float16",
            "range": [(1, 10), (16, 16)],
            "format": "ND",
            "ori_format": "ND",
        },
        {
            "shape": (-1, 16),
            "ori_shape": (-1, 16),
            "dtype": "float16",
            "range": [(1, 10), (16, 16)],
            "format": "ND",
            "ori_format": "ND",
        },
        {
            "shape": (-1, 16),
            "ori_shape": (-1, 16),
            "dtype": "float16",
            "range": [(1, 10), (16, 16)],
            "format": "ND",
            "ori_format": "ND",
        },
        {
            "shape": (-1, 16),
            "ori_shape": (-1, 16),
            "dtype": "float16",
            "range": [(1, 10), (16, 16)],
            "format": "ND",
            "ori_format": "ND",
        }],
    "case_name":
        "test_dynamic_para_check_impl_2",
    "expect":
        RuntimeError,
    "support_expect":
        True
}

# ORG_SHAPE
case3 = {
    "params": [
        {
            "shape": (-1, 16),
            "dtype": "float16",
            "range": [(1, 10), (16, 16)],
            "format": "ND",
            "ori_format": "ND",
        },
        {
            "shape": (-1, 16),
            "ori_shape": (-1, 16),
            "dtype": "float16",
            "range": [(1, 10), (16, 16)],
            "format": "ND",
            "ori_format": "ND",
        },
        {
            "shape": (-1, 16),
            "ori_shape": (-1, 16),
            "dtype": "float16",
            "range": [(1, 10), (16, 16)],
            "format": "ND",
            "ori_format": "ND",
        },
        {
            "shape": (-1, 16),
            "ori_shape": (-1, 16),
            "dtype": "float16",
            "range": [(1, 10), (16, 16)],
            "format": "ND",
            "ori_format": "ND",
        }],
    "case_name":
        "test_dynamic_para_check_impl_3",
    "expect":
        RuntimeError,
    "support_expect":
        True
}

# RANGE
case4 = {
    "params": [
        {
            "shape": (-1, 16),
            "ori_shape": (-1, 16),
            "dtype": "float16",
            "format": "ND",
            "ori_format": "ND",
        },
        {
            "shape": (-1, 16),
            "ori_shape": (-1, 16),
            "dtype": "float16",
            "range": [(1, 10), (16, 16)],
            "format": "ND",
            "ori_format": "ND",
        },
        {
            "shape": (-1, 16),
            "ori_shape": (-1, 16),
            "dtype": "float16",
            "range": [(1, 10), (16, 16)],
            "format": "ND",
            "ori_format": "ND",
        },
        {
            "shape": (-1, 16),
            "ori_shape": (-1, 16),
            "dtype": "float16",
            "range": [(1, 10), (16, 16)],
            "format": "ND",
            "ori_format": "ND",
        }],
    "case_name":
        "test_dynamic_para_check_impl_4",
    "expect":
        RuntimeError,
    "support_expect":
        True
}

# FORMATE
case5 = {
    "params": [
        {
            "shape": (-1, 16),
            "ori_shape": (-1, 16),
            "dtype": "float16",
            "range": [(1, 10), (16, 16)],
            "ori_format": "ND",
        },
        {
            "shape": (-1, 16),
            "ori_shape": (-1, 16),
            "dtype": "float16",
            "range": [(1, 10), (16, 16)],
            "format": "ND",
            "ori_format": "ND",
        },
        {
            "shape": (-1, 16),
            "ori_shape": (-1, 16),
            "dtype": "float16",
            "range": [(1, 10), (16, 16)],
            "format": "ND",
            "ori_format": "ND",
        },
        {
            "shape": (-1, 16),
            "ori_shape": (-1, 16),
            "dtype": "float16",
            "range": [(1, 10), (16, 16)],
            "format": "ND",
            "ori_format": "ND",
        }],
    "case_name":
        "test_dynamic_para_check_impl_5",
    "expect":
        RuntimeError,
    "support_expect":
        True
}

# ORG_FORMATE
case6 = {
    "params": [
        {
            "shape": (-1, 16),
            "ori_shape": (-1, 16),
            "dtype": "float16",
            "range": [(1, 10), (16, 16)],
            "format": "ND",
        },
        {
            "shape": (-1, 16),
            "ori_shape": (-1, 16),
            "dtype": "float16",
            "range": [(1, 10), (16, 16)],
            "format": "ND",
            "ori_format": "ND",
        },
        {
            "shape": (-1, 16),
            "ori_shape": (-1, 16),
            "dtype": "float16",
            "range": [(1, 10), (16, 16)],
            "format": "ND",
            "ori_format": "ND",
        },
        {
            "shape": (-1, 16),
            "ori_shape": (-1, 16),
            "dtype": "float16",
            "range": [(1, 10), (16, 16)],
            "format": "ND",
            "ori_format": "ND",
        }],
    "case_name":
        "test_dynamic_para_check_impl_6",
    "expect":
        RuntimeError,
    "support_expect":
        True
}

# DTYPE
case7 = {
    "params": [
        {
            "shape": (-1, 16),
            "ori_shape": (-1, 16),
            "range": [(1, 10), (16, 16)],
            "format": "ND",
            "ori_format": "ND",
        },
        {
            "shape": (-1, 16),
            "ori_shape": (-1, 16),
            "dtype": "float16",
            "range": [(1, 10), (16, 16)],
            "format": "ND",
            "ori_format": "ND",
        },
        {
            "shape": (-1, 16),
            "ori_shape": (-1, 16),
            "dtype": "float16",
            "range": [(1, 10), (16, 16)],
            "format": "ND",
            "ori_format": "ND",
        },
        {
            "shape": (-1, 16),
            "ori_shape": (-1, 16),
            "dtype": "float16",
            "range": [(1, 10), (16, 16)],
            "format": "ND",
            "ori_format": "ND",
        }],
    "case_name":
        "test_dynamic_para_check_impl_7",
    "expect":
        RuntimeError,
    "support_expect":
        True
}

# INPUT DICT
case8 = {
    "params": [
        [],
        {
            "shape": (-1, 16),
            "ori_shape": (-1, 16),
            "dtype": "float16",
            "range": [(1, 10), (16, 16)],
            "format": "ND",
            "ori_format": "ND",
        },
        {
            "shape": (-1, 16),
            "ori_shape": (-1, 16),
            "dtype": "float16",
            "range": [(1, 10), (16, 16)],
            "format": "ND",
            "ori_format": "ND",
        },
        {
            "shape": (-1, 16),
            "ori_shape": (-1, 16),
            "dtype": "float16",
            "range": [(1, 10), (16, 16)],
            "format": "ND",
            "ori_format": "ND",
        }],
    "case_name":
        "test_dynamic_para_check_impl_8",
    "expect":
        RuntimeError,
    "support_expect":
        True
}

# NO OUTPUT
case9 = {
    "params": [
        {
            "shape": (-1, 16),
            "ori_shape": (-1, 16),
            "dtype": "float16",
            "range": [(1, 10), (16, 16)],
            "format": "ND",
            "ori_format": "ND",
        },
        None,
        {
            "shape": (-1, 16),
            "ori_shape": (-1, 16),
            "dtype": "float16",
            "range": [(1, 10), (16, 16)],
            "format": "ND",
            "ori_format": "ND",
        },
        {
            "shape": (-1, 16),
            "ori_shape": (-1, 16),
            "dtype": "float16",
            "range": [(1, 10), (16, 16)],
            "format": "ND",
            "ori_format": "ND",
        }],
    "case_name":
        "test_dynamic_para_check_impl_9",
    "expect":
        TypeError,
    "support_expect":
        True
}

# SHAPE VALUE ERROR
case10 = {
    "params": [
        {
            "shape": ("-1", 16),
            "ori_shape": (-1, 16),
            "dtype": "float16",
            "range": [(1, 10), (16, 16)],
            "format": "ND",
            "ori_format": "ND",
        },
        {
            "shape": (-1, 16),
            "ori_shape": (-1, 16),
            "dtype": "float16",
            "range": [(1, 10), (16, 16)],
            "format": "ND",
            "ori_format": "ND",
        },
        {
            "shape": (-1, 16),
            "ori_shape": (-1, 16),
            "dtype": "float16",
            "range": [(1, 10), (16, 16)],
            "format": "ND",
            "ori_format": "ND",
        },
        {
            "shape": (-1, 16),
            "ori_shape": (-1, 16),
            "dtype": "float16",
            "range": [(1, 10), (16, 16)],
            "format": "ND",
            "ori_format": "ND",
        }],
    "case_name":
        "test_dynamic_para_check_impl_10",
    "expect":
        RuntimeError,
    "support_expect":
        True
}

# ORG SHAPE VALUE ERROR
case11 = {
    "params": [
        {
            "shape": (-1, 16),
            "ori_shape": ("-1", 16),
            "dtype": "float16",
            "range": [(1, 10), (16, 16)],
            "format": "ND",
            "ori_format": "ND",
        },
        {
            "shape": (-1, 16),
            "ori_shape": (-1, 16),
            "dtype": "float16",
            "range": [(1, 10), (16, 16)],
            "format": "ND",
            "ori_format": "ND",
        },
        {
            "shape": (-1, 16),
            "ori_shape": (-1, 16),
            "dtype": "float16",
            "range": [(1, 10), (16, 16)],
            "format": "ND",
            "ori_format": "ND",
        },
        {
            "shape": (-1, 16),
            "ori_shape": (-1, 16),
            "dtype": "float16",
            "range": [(1, 10), (16, 16)],
            "format": "ND",
            "ori_format": "ND",
        }],
    "case_name":
        "test_dynamic_para_check_impl_11",
    "expect":
        RuntimeError,
    "support_expect":
        True
}

# DTYPE VALUE ERROR
case12 = {
    "params": [
        {
            "shape": (-1, 16),
            "ori_shape": (-1, 16),
            "dtype": "XXX",
            "range": [(1, 10), (16, 16)],
            "format": "ND",
            "ori_format": "ND",
        },
        {
            "shape": (-1, 16),
            "ori_shape": (-1, 16),
            "dtype": "float16",
            "range": [(1, 10), (16, 16)],
            "format": "ND",
            "ori_format": "ND",
        },
        {
            "shape": (-1, 16),
            "ori_shape": (-1, 16),
            "dtype": "float16",
            "range": [(1, 10), (16, 16)],
            "format": "ND",
            "ori_format": "ND",
        },
        {
            "shape": (-1, 16),
            "ori_shape": (-1, 16),
            "dtype": "float16",
            "range": [(1, 10), (16, 16)],
            "format": "ND",
            "ori_format": "ND",
        }],
    "case_name":
        "test_dynamic_para_check_impl_12",
    "expect":
        RuntimeError,
    "support_expect":
        True
}

# RANGE COUNT ERROR
case13 = {
    "params": [
        {
            "shape": (-1, 16),
            "ori_shape": (-1, 16),
            "dtype": "float16",
            "range": [(1, 10), ],
            "format": "ND",
            "ori_format": "ND",
        },
        {
            "shape": (-1, 16),
            "ori_shape": (-1, 16),
            "dtype": "float16",
            "range": [(1, 10), (16, 16)],
            "format": "ND",
            "ori_format": "ND",
        },
        {
            "shape": (-1, 16),
            "ori_shape": (-1, 16),
            "dtype": "float16",
            "range": [(1, 10), (16, 16)],
            "format": "ND",
            "ori_format": "ND",
        },
        {
            "shape": (-1, 16),
            "ori_shape": (-1, 16),
            "dtype": "float16",
            "range": [(1, 10), (16, 16)],
            "format": "ND",
            "ori_format": "ND",
        }],
    "case_name":
        "test_dynamic_para_check_impl_13",
    "expect":
        RuntimeError,
    "support_expect":
        True
}

# RANGE ERROR1
case14 = {
    "params": [
        {
            "shape": (-1, 16),
            "ori_shape": (-1, 16),
            "dtype": "float16",
            "range": [(1, 10), (16, 1)],
            "format": "ND",
            "ori_format": "ND",
        },
        {
            "shape": (-1, 16),
            "ori_shape": (-1, 16),
            "dtype": "float16",
            "range": [(1, 10), (16, 16)],
            "format": "ND",
            "ori_format": "ND",
        },
        {
            "shape": (-1, 16),
            "ori_shape": (-1, 16),
            "dtype": "float16",
            "range": [(1, 10), (16, 16)],
            "format": "ND",
            "ori_format": "ND",
        },
        {
            "shape": (-1, 16),
            "ori_shape": (-1, 16),
            "dtype": "float16",
            "range": [(1, 10), (16, 16)],
            "format": "ND",
            "ori_format": "ND",
        }],
    "case_name":
        "test_dynamic_para_check_impl_14",
    "expect":
        RuntimeError,
    "support_expect":
        True
}

# RANGE ERROR2
case15 = {
    "params": [
        {
            "shape": (-1, 16),
            "ori_shape": (-1, 16),
            "dtype": "float16",
            "range": [(1, 10), (16, "1")],
            "format": "ND",
            "ori_format": "ND",
        },
        {
            "shape": (-1, 16),
            "ori_shape": (-1, 16),
            "dtype": "float16",
            "range": [(1, 10), (16, 16)],
            "format": "ND",
            "ori_format": "ND",
        },
        {
            "shape": (-1, 16),
            "ori_shape": (-1, 16),
            "dtype": "float16",
            "range": [(1, 10), (16, 16)],
            "format": "ND",
            "ori_format": "ND",
        },
        {
            "shape": (-1, 16),
            "ori_shape": (-1, 16),
            "dtype": "float16",
            "range": [(1, 10), (16, 16)],
            "format": "ND",
            "ori_format": "ND",
        }],
    "case_name":
        "test_dynamic_para_check_impl_15",
    "expect":
        RuntimeError,
    "support_expect":
        True
}

# FORMAT
case16 = {
    "params": [
        {
            "shape": (-1, 16),
            "ori_shape": (-1, 16),
            "dtype": "float16",
            "range": [(1, 10), (16, 16)],
            "format": "xxxx",
            "ori_format": "ND",
        },
        {
            "shape": (-1, 16),
            "ori_shape": (-1, 16),
            "dtype": "float16",
            "range": [(1, 10), (16, 16)],
            "format": "ND",
            "ori_format": "ND",
        },
        {
            "shape": (-1, 16),
            "ori_shape": (-1, 16),
            "dtype": "float16",
            "range": [(1, 10), (16, 16)],
            "format": "ND",
            "ori_format": "ND",
        },
        {
            "shape": (-1, 16),
            "ori_shape": (-1, 16),
            "dtype": "float16",
            "range": [(1, 10), (16, 16)],
            "format": "ND",
            "ori_format": "ND",
        }],
    "case_name":
        "test_dynamic_para_check_impl_16",
    "expect":
        RuntimeError,
    "support_expect":
        True
}

# FORMAT
case17 = {
    "params": [
        {
            "shape": (-1, 16),
            "ori_shape": (-1, 16),
            "dtype": "float16",
            "range": [(1, 10), (16, 16)],
            "format": "ND",
            "ori_format": "XXXX",
        },
        {
            "shape": (-1, 16),
            "ori_shape": (-1, 16),
            "dtype": "float16",
            "range": [(1, 10), (16, 16)],
            "format": "ND",
            "ori_format": "ND",
        },
        {
            "shape": (-1, 16),
            "ori_shape": (-1, 16),
            "dtype": "float16",
            "range": [(1, 10), (16, 16)],
            "format": "ND",
            "ori_format": "ND",
        },
        {
            "shape": (-1, 16),
            "ori_shape": (-1, 16),
            "dtype": "float16",
            "range": [(1, 10), (16, 16)],
            "format": "ND",
            "ori_format": "ND",
        }],
    "case_name":
        "test_dynamic_para_check_impl_17",
    "expect":
        RuntimeError,
    "support_expect":
        True
}

# KERNEL NAME ERROR
case18 = {
    "params": [
        {
            "shape": (-1, 16),
            "ori_shape": (-1, 16),
            "dtype": "float16",
            "range": [(1, 10), (16, 16)],
            "format": "ND",
            "ori_format": "ND",
        },
        {
            "shape": (-1, 16),
            "ori_shape": (-1, 16),
            "dtype": "float16",
            "range": [(1, 10), (16, 16)],
            "format": "ND",
            "ori_format": "ND",
        },
        {
            "shape": (-1, 16),
            "ori_shape": (-1, 16),
            "dtype": "float16",
            "range": [(1, 10), (16, 16)],
            "format": "ND",
            "ori_format": "ND",
        },
        {
            "shape": (-1, 16),
            "ori_shape": (-1, 16),
            "dtype": "float16",
            "range": [(1, 10), (16, 16)],
            "format": "ND",
            "ori_format": "ND",
        }],
    "case_name":
        "test_dynamic_para_check_impl_18_*****",
    "expect":
        RuntimeError,
    "support_expect":
        True
}

# KERNEL NAME ERROR
case19 = {
    "params": [
        {
            "shape": (-1, 16),
            "ori_shape": (-1, 16),
            "dtype": "float16",
            "range": [(1, 10), (16, 16)],
            "format": "ND",
            "ori_format": "ND",
        },
        {
            "shape": (-1, 16),
            "ori_shape": (-1, 16),
            "dtype": "float16",
            "range": [(1, 10), (16, 16)],
            "format": "ND",
            "ori_format": "ND",
        },
        {
            "shape": (-1, 16),
            "ori_shape": (-1, 16),
            "dtype": "float16",
            "range": [(1, 10), (16, 16)],
            "format": "ND",
            "ori_format": "ND",
        },
        {
            "shape": (-1, 16),
            "ori_shape": (-1, 16),
            "dtype": "float16",
            "range": [(1, 10), (16, 16)],
            "format": "ND",
            "ori_format": "ND",
        }],
    "case_name":
        "test_dynamic_para_check_impl_19" * 10,
    "expect":
        RuntimeError,
    "support_expect":
        True
}

# DTYPE TYPE ERROR
case20 = {
    "params": [
        {
            "shape": (-1, 16),
            "ori_shape": (-1, 16),
            "dtype": 1,
            "range": [(1, 10), (16, 16)],
            "format": "ND",
            "ori_format": "ND",
        },
        {
            "shape": (-1, 16),
            "ori_shape": (-1, 16),
            "dtype": "float16",
            "range": [(1, 10), (16, 16)],
            "format": "ND",
            "ori_format": "ND",
        },
        {
            "shape": (-1, 16),
            "ori_shape": (-1, 16),
            "dtype": "float16",
            "range": [(1, 10), (16, 16)],
            "format": "ND",
            "ori_format": "ND",
        },
        {
            "shape": (-1, 16),
            "ori_shape": (-1, 16),
            "dtype": "float16",
            "range": [(1, 10), (16, 16)],
            "format": "ND",
            "ori_format": "ND",
        }],
    "case_name":
        "test_dynamic_para_check_impl_20",
    "expect":
        RuntimeError,
    "support_expect":
        True
}

# INPUT NONE
case21 = {
    "params": [
        None, {
            "shape": (-1, 16),
            "ori_shape": (-1, 16),
            "dtype": "float16",
            "range": [(1, 10), (16, 16)],
            "format": "ND",
            "ori_format": "ND", },
        {
            "shape": (-1, 16),
            "ori_shape": (-1, 16),
            "dtype": "float16",
            "range": [(1, 10), (16, 16)],
            "format": "ND",
            "ori_format": "ND",
        },
        {
            "shape": (-1, 16),
            "ori_shape": (-1, 16),
            "dtype": "float16",
            "range": [(1, 10), (16, 16)],
            "format": "ND",
            "ori_format": "ND",
        }],
    "case_name":
        "test_dynamic_para_check_impl_21",
    "expect":
        RuntimeError,
    "support_expect":
        True
}

# OUTPUT NONE
case22 = {
    "params": [
        {
            "shape": (-1, 16),
            "ori_shape": (-1, 16),
            "dtype": "float16",
            "range": [(1, 10), (16, 16)],
            "format": "ND",
            "ori_format": "ND",
        },
        {
            "shape": (-1, 16),
            "ori_shape": (-1, 16),
            "dtype": "float16",
            "range": [(1, 10), (16, 16)],
            "format": "ND",
            "ori_format": "ND",
        },
        None,
        {
            "shape": (-1, 16),
            "ori_shape": (-1, 16),
            "dtype": "float16",
            "range": [(1, 10), (16, 16)],
            "format": "ND",
            "ori_format": "ND",
        }],
    "case_name":
        "test_dynamic_para_check_impl_22",
    "expect":
        RuntimeError,
    "support_expect":
        True
}


# RANGE TYPE
case23 = {
    "params": [
        {
            "shape": (-1, 16),
            "ori_shape": (-1, 16),
            "dtype": "float16",
            "range": 16,
            "format": "ND",
            "ori_format": "ND",
        },
        {
            "shape": (-1, 16),
            "ori_shape": (-1, 16),
            "dtype": "float16",
            "range": [(1, 10), (16, 16)],
            "format": "ND",
            "ori_format": "ND",
        },
        {
            "shape": (-1, 16),
            "ori_shape": (-1, 16),
            "dtype": "float16",
            "range": [(1, 10), (16, 16)],
            "format": "ND",
            "ori_format": "ND",
        },
        {
            "shape": (-1, 16),
            "ori_shape": (-1, 16),
            "dtype": "float16",
            "range": [(1, 10), (16, 16)],
            "format": "ND",
            "ori_format": "ND",
        }],
    "case_name":
        "test_dynamic_para_check_impl_23",
    "expect":
        RuntimeError,
    "support_expect":
        True
}

# RANGE LOW ERROR
case24 = {
    "params": [
        {
            "shape": (-1, 16),
            "ori_shape": (-1, 16),
            "dtype": "float16",
            "range": [("1", 10), (16, 16)],
            "format": "ND",
            "ori_format": "ND",
        },
        {
            "shape": (-1, 16),
            "ori_shape": (-1, 16),
            "dtype": "float16",
            "range": [(1, 10), (16, 16)],
            "format": "ND",
            "ori_format": "ND",
        },
        {
            "shape": (-1, 16),
            "ori_shape": (-1, 16),
            "dtype": "float16",
            "range": [(1, 10), (16, 16)],
            "format": "ND",
            "ori_format": "ND",
        },
        {
            "shape": (-1, 16),
            "ori_shape": (-1, 16),
            "dtype": "float16",
            "range": [(1, 10), (16, 16)],
            "format": "ND",
            "ori_format": "ND",
        }],
    "case_name":
        "test_dynamic_para_check_impl_24",
    "expect":
        RuntimeError,
    "support_expect":
        True
}

# RANGE LOW ERROR
case25 = {
    "params": [
        {
            "shape": (-1, 16),
            "ori_shape": (-1, 16),
            "dtype": "float16",
            "range": [(1, 10, 100), (16, 16)],
            "format": "ND",
            "ori_format": "ND",
        },
        {
            "shape": (-1, 16),
            "ori_shape": (-1, 16),
            "dtype": "float16",
            "range": [(1, 10), (16, 16)],
            "format": "ND",
            "ori_format": "ND",
        },
        {
            "shape": (-1, 16),
            "ori_shape": (-1, 16),
            "dtype": "float16",
            "range": [(1, 10), (16, 16)],
            "format": "ND",
            "ori_format": "ND",
        },
        {
            "shape": (-1, 16),
            "ori_shape": (-1, 16),
            "dtype": "float16",
            "range": [(1, 10), (16, 16)],
            "format": "ND",
            "ori_format": "ND",
        }],
    "case_name":
        "test_dynamic_para_check_impl_25",
    "expect":
        RuntimeError,
    "support_expect":
        True
}

# RANGE LOW ERROR
case26 = {
    "params": [
        {
            "shape": (-1, 2**32),
            "ori_shape": (-1, 2**32),
            "dtype": "float16",
            "range": [(1, 10,), (2**32, 2**32)],
            "format": "ND",
            "ori_format": "ND",
        },
        {
            "shape": (-1, 2**32),
            "ori_shape": (-1, 2**32),
            "dtype": "float16",
            "range": [(1, 10,), (2**32, 2**32)],
            "format": "ND",
            "ori_format": "ND",
        },
        {
            "shape": (-1, 2**32),
            "ori_shape": (-1, 2**32),
            "dtype": "float16",
            "range": [(1, 10,), (2**32, 2**32)],
            "format": "ND",
            "ori_format": "ND",
        },
        {
            "shape": (-1, 2**32),
            "ori_shape": (-1, 2**32),
            "dtype": "float16",
            "range": [(1, 10,), (2**32, 2**32)],
            "format": "ND",
            "ori_format": "ND",
        },],
    "case_name":
        "test_dynamic_para_check_impl_26",
    "expect":
        RuntimeError,
    "support_expect":
        True
}

ut_case.add_case(["all", ], case1)
ut_case.add_case(["all", ], case2)
ut_case.add_case(["all", ], case3)
ut_case.add_case(["all", ], case4)
ut_case.add_case(["all", ], case5)
ut_case.add_case(["all", ], case6)
ut_case.add_case(["all", ], case7)
ut_case.add_case(["all", ], case8)
ut_case.add_case(["all", ], case9)
ut_case.add_case(["all", ], case10)
ut_case.add_case(["all", ], case11)
ut_case.add_case(["all", ], case12)
ut_case.add_case(["all", ], case13)
ut_case.add_case(["all", ], case14)
ut_case.add_case(["all", ], case15)
ut_case.add_case(["all", ], case16)
ut_case.add_case(["all", ], case17)
ut_case.add_case(["all", ], case18)
ut_case.add_case(["all", ], case19)
ut_case.add_case(["all", ], case20)
ut_case.add_case(["all", ], case21)
ut_case.add_case(["all", ], case22)
ut_case.add_case(["all", ], case23)
ut_case.add_case(["all", ], case24)
ut_case.add_case(["all", ], case25)
ut_case.add_case(["all", ], case26)
