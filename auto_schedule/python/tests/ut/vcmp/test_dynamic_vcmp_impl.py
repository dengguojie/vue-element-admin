# # -*- coding:utf-8 -*-
from sch_test_frame.ut import OpUT
from sch_test_frame.utils.op_param_util import cartesian_set_format_dtype
from sch_test_frame.common import precision_info
import numpy as np

import tbe
from tbe import tvm
from tbe.common.utils import shape_util
from tbe.common.register import register_operator


@register_operator("vcmp")
def dsl_dynamic_vcmp(x, y, z, operation, mode, kernel_name="dsl_dynamic_vcmp"):
    input_dtype = x.get("dtype")
    # y can be input or scalar
    schedules, tensors = [], []
    if isinstance(y, dict):
        ins = tbe.dsl.classify([x, y], "elewise")
        for (x, y) in ins:
            with tbe.dsl.compute():
                shape_x, shape_y = shape_util.variable_shape([x, y])
                data1 = tvm.placeholder(shape_x, name='data1', dtype=input_dtype)
                data2 = tvm.placeholder(shape_y, name='data2', dtype=input_dtype)
                res = tbe.dsl.vcmp(data1, data2, operation, mode)

                tensors.append((data1, data2, res))

            with tvm.target.cce():
                sch = tbe.dsl.auto_schedule(res)
            schedules.append(sch)
    else:
        ins = tbe.dsl.classify([x], "elewise")
        y_scalar = tvm.const(y,dtype=input_dtype)
        for (x,) in ins:
            with tbe.dsl.compute():
                shape_x = shape_util.variable_shape([x])[0]
                data1 = tvm.placeholder(shape_x, name='data1', dtype=input_dtype)
                res = tbe.dsl.vcmp(data1, y_scalar, operation, mode)

                tensors.append((data1, res))

            with tvm.target.cce():
                sch = tbe.dsl.auto_schedule(res)
            schedules.append(sch)

    config = {"name": kernel_name, "tensor_list": tensors}
    tbe.dsl.build(schedules, config)


ut_case = OpUT("vcmp", "vcmp.test_dynamic_vcmp_impl", "dsl_dynamic_vcmp")

case1 = {
    "params": [{
        "shape": (-1, -1),
        "dtype": "float16",
        "range": [(1, None), (1, None)]
    }, {
        "shape": (-1, -1),
        "dtype": "float16",
        "range": [(1, None), (1, None)]
    }, {
        "shape": (-1, -1),
        "dtype": "uint8",
        "range": [(1, None), (1, None)]
    },
    "eq", "bool"
    ],
    "case_name":
        "test_dync_vcmp_1",
    "expect":
        "success",
    "support_expect":
        True
}

case2 = {
    "params": [{
        "shape": (-1, -1),
        "dtype": "float32",
        "range": [(1, None), (1, None)]
    }, {
        "shape": (-1, -1),
        "dtype": "float32",
        "range": [(1, None), (1, None)]
    }, {
        "shape": (-1, -1),
        "dtype": "uint8",
        "range": [(1, None), (1, None)]
    },
    "ne", "bit"
    ],
    "case_name":
        "test_dync_vcmp_2",
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
        "dtype": "float16",
        "range": [(1, None), (1, None)]
    }, {
        "shape": (-1, -1),
        "dtype": "uint8",
        "range": [(1, None), (1, None)]
    },
    "lt", "bool"
    ],
    "case_name":
        "test_dync_vcmp_3",
    "expect":
        "success",
    "support_expect":
        True
}

case4 = {
    "params": [{
        "shape": (-1, -1),
        "dtype": "float32",
        "range": [(1, None), (1, None)]
    }, {
        "shape": (-1, -1),
        "dtype": "float32",
        "range": [(1, None), (1, None)]
    }, {
        "shape": (-1, -1),
        "dtype": "uint8",
        "range": [(1, None), (1, None)]
    },
    "gt", "bit"
    ],
    "case_name":
        "test_dync_vcmp_4",
    "expect":
        "success",
    "support_expect":
        True
}

case5 = {
    "params": [{
        "shape": (-1, -1),
        "dtype": "float16",
        "range": [(1, None), (1, None)]
    }, {
        "shape": (-1, -1),
        "dtype": "float16",
        "range": [(1, None), (1, None)]
    }, {
        "shape": (-1, -1),
        "dtype": "uint8",
        "range": [(1, None), (1, None)]
    },
    "ge", "bool"
    ],
    "case_name":
        "test_dync_vcmp_5",
    "expect":
        "success",
    "support_expect":
        True
}

case6 = {
    "params": [{
        "shape": (-1, -1),
        "dtype": "float32",
        "range": [(1, None), (1, None)]
    }, {
        "shape": (-1, -1),
        "dtype": "float32",
        "range": [(1, None), (1, None)]
    }, {
        "shape": (-1, -1),
        "dtype": "uint8",
        "range": [(1, None), (1, None)]
    },
    "le", "bit"
    ],
    "case_name":
        "test_dync_vcmp_6",
    "expect":
        "success",
    "support_expect":
        True
}

case7 = {
    "params": [{
        "shape": (-1, -1),
        "dtype": "float16",
        "range": [(1, None), (1, None)]
    }, 0.5,
    {
        "shape": (-1, -1),
        "dtype": "uint8",
        "range": [(1, None), (1, None)]
    },
    "ge", "bool"
    ],
    "case_name":
        "test_dync_vcmp_7",
    "expect":
        "success",
    "support_expect":
        True
}

case8 = {
    "params": [{
        "shape": (-1, -1),
        "dtype": "float32",
        "range": [(1, None), (1, None)]
    }, 0.25,
    {
        "shape": (-1, -1),
        "dtype": "uint8",
        "range": [(1, None), (1, None)]
    },
    "le", "bit"
    ],
    "case_name":
        "test_dync_vcmp_8",
    "expect":
        "success",
    "support_expect":
        True
}

ut_case.add_case(["Ascend910A", "Ascend710"], case1)
ut_case.add_case(["Ascend910A"], case2)
ut_case.add_case(["Ascend910A", "Ascend710"], case3)
ut_case.add_case(["Ascend910A"], case4)
ut_case.add_case(["Ascend910A", "Ascend710"], case5)
ut_case.add_case(["Ascend910A"], case6)
ut_case.add_case(["Ascend910A", "Ascend710"], case7)
ut_case.add_case(["Ascend910A"], case8)


def calc_expect_func(x, y, z, operation, mode):
    x_value = np.array(x.get("value")).reshape(-1)
    x_run_shape = list(x.get("run_shape"))
    x_size = len(x_value)
    input_dtype = x.get("dtype")
    if isinstance(y, dict):
        y_value = np.array(y.get("value")).reshape(-1)
    else:
        y_value = y * np.ones((x_size,), input_dtype)
    if mode == "bool":
        condition = np.zeros((x_size,), np.uint8)
        if operation == "eq":
            for i in range(x_size):
                output_flag = 1 if x_value[i] == y_value[i] else 0
                condition[i] = output_flag
        elif operation == "ge":
            for i in range(x_size):
                output_flag = 1 if x_value[i] >= y_value[i] else 0
                condition[i] = output_flag
        elif operation == "le":
            for i in range(x_size):
                output_flag = 1 if x_value[i] <= y_value[i] else 0
                condition[i] = output_flag
        elif operation == "gt":
            for i in range(x_size):
                output_flag = 1 if x_value[i] > y_value[i] else 0
                condition[i] = output_flag
        elif operation == "lt":
            for i in range(x_size):
                output_flag = 1 if x_value[i] < y_value[i] else 0
                condition[i] = output_flag
        else:
            for i in range(x_size):
                output_flag = 1 if x_value[i] != y_value[i] else 0
                condition[i] = output_flag
        return condition.reshape(x_run_shape)
    else:
        output = np.zeros((x_size,), np.uint8)
        condition = np.zeros((x_size,), np.uint8)
        index = 0
        tmp_condition = 0
        if operation == "eq":
            for i in range(x_size):
                output_flag = 1 if x_value[i] == y_value[i] else 0
                condition[i] = output_flag
                inner_index = i % 8
                tmp_condition += output_flag*np.power(2, inner_index)
                if inner_index == 7:
                    output[index] = tmp_condition
                    tmp_condition = 0
                    index += 1
        elif operation == "ge":
            for i in range(x_size):
                output_flag = 1 if x_value[i] >= y_value[i] else 0
                condition[i] = output_flag
                inner_index = i % 8
                tmp_condition += output_flag*np.power(2, inner_index)
                if inner_index == 7:
                    output[index] = tmp_condition
                    tmp_condition = 0
                    index += 1
        elif operation == "le":
            for i in range(x_size):
                output_flag = 1 if x_value[i] <= y_value[i] else 0
                condition[i] = output_flag
                inner_index = i % 8
                tmp_condition += output_flag*np.power(2, inner_index)
                if inner_index == 7:
                    output[index] = tmp_condition
                    tmp_condition = 0
                    index += 1
        elif operation == "gt":
            for i in range(x_size):
                output_flag = 1 if x_value[i] > y_value[i] else 0
                condition[i] = output_flag
                inner_index = i % 8
                tmp_condition += output_flag*np.power(2, inner_index)
                if inner_index == 7:
                    output[index] = tmp_condition
                    tmp_condition = 0
                    index += 1
        elif operation == "lt":
            for i in range(x_size):
                output_flag = 1 if x_value[i] < y_value[i] else 0
                condition[i] = output_flag
                inner_index = i % 8
                tmp_condition += output_flag*np.power(2, inner_index)
                if inner_index == 7:
                    output[index] = tmp_condition
                    tmp_condition = 0
                    index += 1
        else:
            for i in range(x_size):
                output_flag = 1 if x_value[i] != y_value[i] else 0
                condition[i] = output_flag
                inner_index = i % 8
                tmp_condition += output_flag*np.power(2, inner_index)
                if inner_index == 7:
                    output[index] = tmp_condition
                    tmp_condition = 0
                    index += 1
        return output.reshape(x_run_shape)

ut_case.add_precision_case(
    ["Ascend910A", "Ascend710"], {
        "params": [
            {
                "shape": (-1, -1),
                "dtype": "float16",
                "range": [(1, 200), (1, None)],
                "run_shape": (2, 256),
                "param_type": "input"
            },
            {
                "shape": (-1, -1),
                "dtype": "float16",
                "range": [(1, 200), (1, None)],
                "run_shape": (2, 256),
                "param_type": "input"
            },
            {
                "shape": (-1, -1),
                "dtype": "uint8",
                "range": [(1, 200), (1, None)],
                "run_shape": (2, 256),
                "param_type": "output"
            }, "gt", "bit"
        ],
        "calc_expect_func": calc_expect_func,
        "precision_standard": precision_info.PrecisionStandard(0.001, 0.001),
        "case_name": "test_dync_vcmp_prec_01"
    })
