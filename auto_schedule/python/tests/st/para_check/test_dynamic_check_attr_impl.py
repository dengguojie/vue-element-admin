import numpy as np
import tbe
from sch_test_frame.common import precision_info
from sch_test_frame.ut import OpUT
from tbe import tvm
from tbe.common.register import register_operator
from tbe.common.utils import shape_util
from tbe.dsl import classify
from tbe.common.utils import para_check


@register_operator("check_attr")
@para_check.check_op_params(para_check.REQUIRED_INPUT, para_check.REQUIRED_OUTPUT,
                            float, para_check.REQUIRED_ATTR_STR, para_check.REQUIRED_ATTR_INT,
                            para_check.REQUIRED_ATTR_FLOAT, para_check.REQUIRED_ATTR_BOOL,
                            para_check.REQUIRED_ATTR_TYPE, para_check.REQUIRED_ATTR_LIST_INT,
                            para_check.REQUIRED_ATTR_LIST_FLOAT, para_check.REQUIRED_ATTR_LIST_BOOL,
                            para_check.REQUIRED_ATTR_LIST_LIST_INT, para_check.DYNAMIC_INPUT,
                            para_check.DYNAMIC_OUTPUT, para_check.KERNEL_NAME)
def dsl_dync_check_attr(x, y,
                        value, attr_str, attr_int,
                        attr_float, attr_bool,
                        attr_type, attr_list_int,
                        attr_list_float, attr_list_bool,
                        attr_list_list_int, dynamic_input,
                        dynamic_output, kernel_name="dsl_dync_check_attr"):
    input_dtype = x.get("dtype")

    ins = classify([x], "elewise")
    schedules, tensors = [], []

    for (x,) in ins:
        with tbe.dsl.compute():
            shape_x = shape_util.variable_shape([x])[0]
            data1 = tvm.placeholder(shape_x, name='data1', dtype=input_dtype)
            res = tbe.dsl.vadds(data1, tvm.const(value, dtype=input_dtype))

            tensors.append((data1, res))

        with tvm.target.cce():
            sch = tbe.dsl.auto_schedule(res)
        schedules.append(sch)

    config = {"name": kernel_name, "tensor_list": tensors}
    tbe.dsl.build(schedules, config)


ut_case = OpUT("check_attr", "para_check.test_dynamic_check_attr_impl", "dsl_dync_check_attr")

# attr str type error
case1 = {
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
        2.0,
        9,
        1,
        1.0,
        True,
        10,
        [1, 1],
        [1.0, 1.5],
        [False, False],
        [[1, 1], [1, 1], ],
        [
            {
                "shape": (-1, 16),
                "ori_shape": (-1, 16),
                "dtype": "float16",
                "range": [(1, 10), (16, 16)],
                "format": "ND",
                "ori_format": "ND",
            },
        ],
        [
            {
                "shape": (-1, 16),
                "ori_shape": (-1, 16),
                "dtype": "float16",
                "range": [(1, 10), (16, 16)],
                "format": "ND",
                "ori_format": "ND",
            },
        ],

    ],
    "expect":
        RuntimeError,
    "support_expect":
        True
}

# attr None
case2 = {
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
        2.0,
        "9",
        1,
        None,
        True,
        10,
        [1, 1],
        [1.0, 1.5],
        [False, False],
        [[1, 1], [1, 1], ],
        [
            {
                "shape": (-1, 16),
                "ori_shape": (-1, 16),
                "dtype": "float16",
                "range": [(1, 10), (16, 16)],
                "format": "ND",
                "ori_format": "ND",
            },
        ],
        [
            {
                "shape": (-1, 16),
                "ori_shape": (-1, 16),
                "dtype": "float16",
                "range": [(1, 10), (16, 16)],
                "format": "ND",
                "ori_format": "ND",
            },
        ],
    ],
    "expect":
        RuntimeError,
    "support_expect":
        True
}

# attr int type
case3 = {
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
        1.0,
        "attr_int",
        1.0,
        2.1,
        True,
        10,
        [1, 1],
        [1.0, 1.5],
        [False, False],
        [[1, 1], [1, 1], ],
        [
            {
                "shape": (-1, 16),
                "ori_shape": (-1, 16),
                "dtype": "float16",
                "range": [(1, 10), (16, 16)],
                "format": "ND",
                "ori_format": "ND",
            },
        ],
        [
            {
                "shape": (-1, 16),
                "ori_shape": (-1, 16),
                "dtype": "float16",
                "range": [(1, 10), (16, 16)],
                "format": "ND",
                "ori_format": "ND",
            },
        ],
    ],
    "expect":
        RuntimeError,
    "support_expect":
        True
}

# attr float type
case4 = {
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
        1.0,
        "str",
        10,
        "xxxx",
        True,
        10,
        [1, 1],
        [1.0, 1.5],
        [False, False],
        [[1, 1], [1, 1], ],
        [
            {
                "shape": (-1, 16),
                "ori_shape": (-1, 16),
                "dtype": "float16",
                "range": [(1, 10), (16, 16)],
                "format": "ND",
                "ori_format": "ND",
            },
        ],
        [
            {
                "shape": (-1, 16),
                "ori_shape": (-1, 16),
                "dtype": "float16",
                "range": [(1, 10), (16, 16)],
                "format": "ND",
                "ori_format": "ND",
            },
        ],
    ],
    "expect":
        RuntimeError,
    "support_expect":
        True
}

# attr float value
case5 = {
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
        1.0,
        "str",
        10,
        float("inf"),
        True,
        10,
        [1, 1],
        [1.0, 1.5],
        [False, False],
        [[1, 1], [1, 1], ],
        [
            {
                "shape": (-1, 16),
                "ori_shape": (-1, 16),
                "dtype": "float16",
                "range": [(1, 10), (16, 16)],
                "format": "ND",
                "ori_format": "ND",
            },
        ],
        [
            {
                "shape": (-1, 16),
                "ori_shape": (-1, 16),
                "dtype": "float16",
                "range": [(1, 10), (16, 16)],
                "format": "ND",
                "ori_format": "ND",
            },
        ],
    ],
    "expect":
        RuntimeError,
    "support_expect":
        True
}

# attr bool value
case6 = {
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
        1.0,
        "str",
        10,
        2.1,
        10,
        10,
        [1, 1],
        [1.0, 1.5],
        [False, False],
        [[1, 1], [1, 1], ],
        [
            {
                "shape": (-1, 16),
                "ori_shape": (-1, 16),
                "dtype": "float16",
                "range": [(1, 10), (16, 16)],
                "format": "ND",
                "ori_format": "ND",
            },
        ],
        [
            {
                "shape": (-1, 16),
                "ori_shape": (-1, 16),
                "dtype": "float16",
                "range": [(1, 10), (16, 16)],
                "format": "ND",
                "ori_format": "ND",
            },
        ],
    ],
    "expect":
        RuntimeError,
    "support_expect":
        True
}

# attr type
case7 = {
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
        1.0,
        "str",
        10,
        2.1,
        True,
        "uint81",
        [1, 1],
        [1.0, 1.5],
        [False, False],
        [[1, 1], [1, 1], ],
        [
            {
                "shape": (-1, 16),
                "ori_shape": (-1, 16),
                "dtype": "float16",
                "range": [(1, 10), (16, 16)],
                "format": "ND",
                "ori_format": "ND",
            },
        ],
        [
            {
                "shape": (-1, 16),
                "ori_shape": (-1, 16),
                "dtype": "float16",
                "range": [(1, 10), (16, 16)],
                "format": "ND",
                "ori_format": "ND",
            },
        ],
    ],
    "expect":
        RuntimeError,
    "support_expect":
        True
}

# attr list not list
case8 = {
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
        1.0,
        "str",
        10,
        2.1,
        True,
        "float16",
        "[1, 1]",
        [1.0, 1.5],
        [False, False],
        [[1, 1], [1, 1], ],
        [
            {
                "shape": (-1, 16),
                "ori_shape": (-1, 16),
                "dtype": "float16",
                "range": [(1, 10), (16, 16)],
                "format": "ND",
                "ori_format": "ND",
            },
        ],
        [
            {
                "shape": (-1, 16),
                "ori_shape": (-1, 16),
                "dtype": "float16",
                "range": [(1, 10), (16, 16)],
                "format": "ND",
                "ori_format": "ND",
            },
        ],
    ],
    "expect":
        RuntimeError,
    "support_expect":
        True
}

# attr int list
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
        {
            "shape": (-1, 16),
            "ori_shape": (-1, 16),
            "dtype": "float16",
            "range": [(1, 10), (16, 16)],
            "format": "ND",
            "ori_format": "ND",
        },
        1.0,
        "str",
        10,
        2.1,
        True,
        "float16",
        [1, 1.0],
        [1.0, 1.5],
        [False, False],
        [[1, 1], [1, 1], ],
        [
            {
                "shape": (-1, 16),
                "ori_shape": (-1, 16),
                "dtype": "float16",
                "range": [(1, 10), (16, 16)],
                "format": "ND",
                "ori_format": "ND",
            },
        ],
        [
            {
                "shape": (-1, 16),
                "ori_shape": (-1, 16),
                "dtype": "float16",
                "range": [(1, 10), (16, 16)],
                "format": "ND",
                "ori_format": "ND",
            },
        ],
    ],
    "expect":
        RuntimeError,
    "support_expect":
        True
}

# attr float list
case10 = {
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
        1.0,
        "str",
        10,
        2.1,
        True,
        "float16",
        [1, 1],
        [1, 1.5],
        [False, False],
        [[1, 1], [1, 1], ],
        [
            {
                "shape": (-1, 16),
                "ori_shape": (-1, 16),
                "dtype": "float16",
                "range": [(1, 10), (16, 16)],
                "format": "ND",
                "ori_format": "ND",
            },
        ],
        [
            {
                "shape": (-1, 16),
                "ori_shape": (-1, 16),
                "dtype": "float16",
                "range": [(1, 10), (16, 16)],
                "format": "ND",
                "ori_format": "ND",
            },
        ],
    ],
    "expect":
        RuntimeError,
    "support_expect":
        True
}

# attr bool list
case11 = {
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
        1.0,
        "str",
        10,
        2.1,
        True,
        "float16",
        [1, 1],
        [1.0, 1.5],
        [1, False],
        [[1, 1], [1, 1], ],
        [
            {
                "shape": (-1, 16),
                "ori_shape": (-1, 16),
                "dtype": "float16",
                "range": [(1, 10), (16, 16)],
                "format": "ND",
                "ori_format": "ND",
            },
        ],
        [
            {
                "shape": (-1, 16),
                "ori_shape": (-1, 16),
                "dtype": "float16",
                "range": [(1, 10), (16, 16)],
                "format": "ND",
                "ori_format": "ND",
            },
        ],
    ],
    "expect":
        RuntimeError,
    "support_expect":
        True
}

# attr int list list
case12 = {
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
        1.0,
        "str",
        10,
        2.1,
        True,
        "float16",
        [1, 1],
        [1.0, 1.5],
        [True, False],
        [["1", 1], [1, 1], ],
        [
            {
                "shape": (-1, 16),
                "ori_shape": (-1, 16),
                "dtype": "float16",
                "range": [(1, 10), (16, 16)],
                "format": "ND",
                "ori_format": "ND",
            },
        ],
        [
            {
                "shape": (-1, 16),
                "ori_shape": (-1, 16),
                "dtype": "float16",
                "range": [(1, 10), (16, 16)],
                "format": "ND",
                "ori_format": "ND",
            },
        ],
    ],
    "expect":
        RuntimeError,
    "support_expect":
        True
}

# attr int list not list
case13 = {
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
        1.0,
        "str",
        10,
        2.1,
        True,
        "float16",
        [1, 1],
        [1.0, 1.5],
        [True, False],
        ["1", [1, 1], ],
        [
            {
                "shape": (-1, 16),
                "ori_shape": (-1, 16),
                "dtype": "float16",
                "range": [(1, 10), (16, 16)],
                "format": "ND",
                "ori_format": "ND",
            },
        ],
        [
            {
                "shape": (-1, 16),
                "ori_shape": (-1, 16),
                "dtype": "float16",
                "range": [(1, 10), (16, 16)],
                "format": "ND",
                "ori_format": "ND",
            },
        ],
    ],
    "expect":
        RuntimeError,
    "support_expect":
        True
}

# dynamic inputs type
case14 = {
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
        1.0,
        "str",
        10,
        2.1,
        True,
        "float16",
        [1, 1],
        [1.0, 1.5],
        [True, False],
        [[1, 1], [1, 1], ],
        "dynamic inputs",
        [
            {
                "shape": (-1, 16),
                "ori_shape": (-1, 16),
                "dtype": "float16",
                "range": [(1, 10), (16, 16)],
                "format": "ND",
                "ori_format": "ND",
            },
        ],
    ],
    "expect":
        RuntimeError,
    "support_expect":
        True
}

# dynamic inputs no shape
case15 = {
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
        1.0,
        "str",
        10,
        2.1,
        True,
        "float16",
        [1, 1],
        [1.0, 1.5],
        [True, False],
        [[1, 1], [1, 1], ],
        [
            {
                "dtype": "float16",
                "range": [(1, 10), (16, 16)],
                "format": "ND",
                "ori_format": "ND",
            },
        ],
        [
            {
                "shape": (-1, 16),
                "ori_shape": (-1, 16),
                "dtype": "float16",
                "range": [(1, 10), (16, 16)],
                "format": "ND",
                "ori_format": "ND",
            },
        ],
    ],
    "expect":
        RuntimeError,
    "support_expect":
        True
}

# dynamic inputs none
case16 = {
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
        1.0,
        "str",
        10,
        2.1,
        True,
        "float16",
        [1, 1],
        [1.0, 1.5],
        [True, False],
        [[1, 1], [1, 1], ],
        [],
        [
            {
                "shape": (-1, 16),
                "ori_shape": (-1, 16),
                "dtype": "float16",
                "range": [(1, 10), (16, 16)],
                "format": "ND",
                "ori_format": "ND",
            },
        ],
    ],
    "expect":
        RuntimeError,
    "support_expect":
        True
}

# dynamic inputs str
case17 = {
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
        1.0,
        "str",
        10,
        2.1,
        True,
        "float16",
        [1, 1],
        [1.0, 1.5],
        [True, False],
        [[1, 1], [1, 1], ],
        [
            {
                "shape": (-1, 16),
                "ori_shape": (-1, 16),
                "dtype": "float16",
                "range": [(1, 10), (16, 16)],
                "format": "ND",
                "ori_format": "ND",
            },
        ],
        "outputs",
    ],
    "expect":
        RuntimeError,
    "support_expect":
        True
}

# dynamic inputs None
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
        1.0,
        "str",
        10,
        2.1,
        True,
        "float16",
        [1, 1],
        [1.0, 1.5],
        [True, False],
        [[1, 1], [1, 1], ],
        [
            {
                "shape": (-1, 16),
                "ori_shape": (-1, 16),
                "dtype": "float16",
                "range": [(1, 10), (16, 16)],
                "format": "ND",
                "ori_format": "ND",
            },
        ],
        [],
    ],
    "expect":
        RuntimeError,
    "support_expect":
        True
}

# dynamic inputs no shape
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
        1.0,
        "str",
        10,
        2.1,
        True,
        "float16",
        [1, 1],
        [1.0, 1.5],
        [True, False],
        [[1, 1], [1, 1], ],
        [
            {
                "shape": (-1, 16),
                "ori_shape": (-1, 16),
                "dtype": "float16",
                "range": [(1, 10), (16, 16)],
                "format": "ND",
                "ori_format": "ND",
            },
        ],
        [
            {
                "dtype": "float16",
                "range": [(1, 10), (16, 16)],
                "format": "ND",
                "ori_format": "ND",
            },
        ],
    ],
    "expect":
        RuntimeError,
    "support_expect":
        True
}

ut_case.add_case(["Ascend910A", "Ascend310"], case1)
ut_case.add_case(["Ascend910A", "Ascend310"], case2)
ut_case.add_case(["Ascend910A", "Ascend310"], case3)
ut_case.add_case(["Ascend910A", "Ascend310"], case4)
ut_case.add_case(["Ascend910A", "Ascend310"], case5)
ut_case.add_case(["Ascend910A", "Ascend310"], case6)
ut_case.add_case(["Ascend910A", "Ascend310"], case7)
ut_case.add_case(["Ascend910A", "Ascend310"], case8)
ut_case.add_case(["Ascend910A", "Ascend310"], case9)
ut_case.add_case(["Ascend910A", "Ascend310"], case10)
ut_case.add_case(["Ascend910A", "Ascend310"], case11)
ut_case.add_case(["Ascend910A", "Ascend310"], case12)
ut_case.add_case(["Ascend910A", "Ascend310"], case13)
ut_case.add_case(["Ascend910A", "Ascend310"], case14)
ut_case.add_case(["Ascend910A", "Ascend310"], case15)
ut_case.add_case(["Ascend910A", "Ascend310"], case16)
ut_case.add_case(["Ascend910A", "Ascend310"], case17)
ut_case.add_case(["Ascend910A", "Ascend310"], case18)
ut_case.add_case(["Ascend910A", "Ascend310"], case19)
