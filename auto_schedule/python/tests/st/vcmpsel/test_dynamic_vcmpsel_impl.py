# # -*- coding:utf-8 -*-
from sch_test_frame.ut import OpUT
from sch_test_frame.utils.op_param_util import cartesian_set_format_dtype
from sch_test_frame.common import precision_info
import numpy as np

import tbe
from tbe import tvm
from tbe.common.utils import shape_util
from tbe.common.register import register_operator


@register_operator("vcmpsel")
def dsl_dynamic_vcmpsel(x1, x2, y1, y2, z, operation, kernel_name="dsl_dynamic_vcmpsel"):
    input_dtype = x1.get("dtype")
    # x1 must be tensor, x2, y1, y2 can be tensor or scalar or None
    if x2 is None:
        x2 = 2.0
    
    if y1 is None:
        y1 = x1
    
    if y2 is None:
        if isinstance(x2, dict):
            y2 = x2
        else:
            y2 = 0.0
    schedules, tensors = [], []
    if isinstance(x2, dict) and isinstance(y1, dict) and isinstance(y2, dict):
        ins = tbe.dsl.classify([x1, x2, y1, y2], "elewise")
        for (x1, x2, y1, y2) in ins:
            with tbe.dsl.compute():
                shape_x1, shape_x2, shape_y1, shape_y2 = shape_util.variable_shape([x1, x2, y1, y2])
                data1 = tvm.placeholder(shape_x1, name='data1', dtype=input_dtype)
                data2 = tvm.placeholder(shape_x2, name='data2', dtype=input_dtype)
                data3 = tvm.placeholder(shape_y1, name='data3', dtype=input_dtype)
                data4 = tvm.placeholder(shape_y2, name='data4', dtype=input_dtype)
                res = tbe.dsl.vcmpsel(data1, data2, operation, data3, data4)

                tensors.append((data1, data2, data3, data4, res))

            with tvm.target.cce():
                sch = tbe.dsl.auto_schedule(res)
            schedules.append(sch)

    elif isinstance(x2, dict) and isinstance(y1, dict) and not isinstance(y2, dict):
        ins = tbe.dsl.classify([x1, x2, y1], "elewise")
        for (x1, x2, y1) in ins:
            with tbe.dsl.compute():
                shape_x1, shape_x2, shape_y1 = shape_util.variable_shape([x1, x2, y1])
                data1 = tvm.placeholder(shape_x1, name='data1', dtype=input_dtype)
                data2 = tvm.placeholder(shape_x2, name='data2', dtype=input_dtype)
                data3 = tvm.placeholder(shape_y1, name='data3', dtype=input_dtype)
                res = tbe.dsl.vcmpsel(data1, data2, operation, data3, y2)

                tensors.append((data1, data2, data3, res))

            with tvm.target.cce():
                sch = tbe.dsl.auto_schedule(res)
            schedules.append(sch)
    
    elif isinstance(x2, dict) and not isinstance(y1, dict) and isinstance(y2, dict):
        ins = tbe.dsl.classify([x1, x2, y2], "elewise")
        for (x1, x2, y2) in ins:
            with tbe.dsl.compute():
                shape_x1, shape_x2, shape_y2 = shape_util.variable_shape([x1, x2, y2])
                data1 = tvm.placeholder(shape_x1, name='data1', dtype=input_dtype)
                data2 = tvm.placeholder(shape_x2, name='data2', dtype=input_dtype)
                data3 = tvm.placeholder(shape_y2, name='data3', dtype=input_dtype)
                res = tbe.dsl.vcmpsel(data1, data2, operation, y1, data3)

                tensors.append((data1, data2, data3, res))

            with tvm.target.cce():
                sch = tbe.dsl.auto_schedule(res)
            schedules.append(sch)
    
    elif not isinstance(x2, dict) and isinstance(y1, dict) and isinstance(y2, dict):
        ins = tbe.dsl.classify([x1, y1, y2], "elewise")
        for (x1, y1, y2) in ins:
            with tbe.dsl.compute():
                shape_x1, shape_y1, shape_y2 = shape_util.variable_shape([x1, y1, y2])
                data1 = tvm.placeholder(shape_x1, name='data1', dtype=input_dtype)
                data2 = tvm.placeholder(shape_y1, name='data2', dtype=input_dtype)
                data3 = tvm.placeholder(shape_y2, name='data3', dtype=input_dtype)
                res = tbe.dsl.vcmpsel(data1, x2, operation, data2, data3)

                tensors.append((data1, data2, data3, res))

            with tvm.target.cce():
                sch = tbe.dsl.auto_schedule(res)
            schedules.append(sch)
    
    elif not isinstance(x2, dict) and not isinstance(y1, dict) and isinstance(y2, dict):
        ins = tbe.dsl.classify([x1, y2], "elewise")
        for (x1, y2) in ins:
            with tbe.dsl.compute():
                shape_x1, shape_y2 = shape_util.variable_shape([x1, y2])
                data1 = tvm.placeholder(shape_x1, name='data1', dtype=input_dtype)
                data2 = tvm.placeholder(shape_y2, name='data2', dtype=input_dtype)
                res = tbe.dsl.vcmpsel(data1, x2, operation, y1, data2)

                tensors.append((data1, data2, res))

            with tvm.target.cce():
                sch = tbe.dsl.auto_schedule(res)
            schedules.append(sch)
    
    elif not isinstance(x2, dict) and isinstance(y1, dict) and not isinstance(y2, dict):
        ins = tbe.dsl.classify([x1, y1], "elewise")
        for (x1, y1) in ins:
            with tbe.dsl.compute():
                shape_x1, shape_y1 = shape_util.variable_shape([x1, y1])
                data1 = tvm.placeholder(shape_x1, name='data1', dtype=input_dtype)
                data2 = tvm.placeholder(shape_y1, name='data2', dtype=input_dtype)
                res = tbe.dsl.vcmpsel(data1, x2, operation, data2, y2)

                tensors.append((data1, data2, res))

            with tvm.target.cce():
                sch = tbe.dsl.auto_schedule(res)
            schedules.append(sch)
    
    elif isinstance(x2, dict) and not isinstance(y1, dict) and not isinstance(y2, dict):
        ins = tbe.dsl.classify([x1, x2], "elewise")
        for (x1, x2) in ins:
            with tbe.dsl.compute():
                shape_x1, shape_x2 = shape_util.variable_shape([x1, x2])
                data1 = tvm.placeholder(shape_x1, name='data1', dtype=input_dtype)
                data2 = tvm.placeholder(shape_x2, name='data2', dtype=input_dtype)
                res = tbe.dsl.vcmpsel(data1, data2, operation, y1, y2)

                tensors.append((data1, data2, res))

            with tvm.target.cce():
                sch = tbe.dsl.auto_schedule(res)
            schedules.append(sch)

    else:
        ins = tbe.dsl.classify([x1], "elewise")
        for (x1,) in ins:
            with tbe.dsl.compute():
                shape_x1 = shape_util.variable_shape([x1])[0]
                data1 = tvm.placeholder(shape_x1, name='data1', dtype=input_dtype)
                res = tbe.dsl.vcmpsel(data1, x2, operation, y1, y2)

                tensors.append((data1, res))

            with tvm.target.cce():
                sch = tbe.dsl.auto_schedule(res)
            schedules.append(sch)

    config = {"name": kernel_name, "tensor_list": tensors}
    tbe.dsl.build(schedules, config)


ut_case = OpUT("vcmpsel", "vcmpsel.test_dynamic_vcmpsel_impl", "dsl_dynamic_vcmpsel")

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
        "dtype": "float16",
        "range": [(1, None), (1, None)]
    }, {
        "shape": (-1, -1),
        "dtype": "float16",
        "range": [(1, None), (1, None)]
    }, {
        "shape": (-1, -1),
        "dtype": "float16",
        "range": [(1, None), (1, None)]
    }, "eq"
    ],
    "case_name":
        "test_dync_vcmpsel_1",
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
    }, None, 
    {
        "shape": (-1, -1),
        "dtype": "float32",
        "range": [(1, None), (1, None)]
    }, {
        "shape": (-1, -1),
        "dtype": "float32",
        "range": [(1, None), (1, None)]
    }, {
        "shape": (-1, -1),
        "dtype": "float32",
        "range": [(1, None), (1, None)]
    }, "ge"
    ],
    "case_name":
        "test_dync_vcmpsel_2",
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
        "dtype": "float16",
        "range": [(1, None), (1, None)]
    }, {
        "shape": (-1, -1),
        "dtype": "float16",
        "range": [(1, None), (1, None)]
    }, {
        "shape": (-1, -1),
        "dtype": "float16",
        "range": [(1, None), (1, None)]
    }, "gt"
    ],
    "case_name":
        "test_dync_vcmpsel_3",
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
    }, 1.0, 2.0, 3.0, 
    {
        "shape": (-1, -1),
        "dtype": "float32",
        "range": [(1, None), (1, None)]
    },
    "le"
    ],
    "case_name":
        "test_dync_vcmpsel_4",
    "expect":
        "success",
    "support_expect":
        True
}

ut_case.add_case(["Ascend910A"], case1)
ut_case.add_case(["Ascend910A"], case2)
ut_case.add_case(["Ascend910A"], case3)
ut_case.add_case(["Ascend910A"], case4)

def calc_expect_func(x1, x2, y1, y2, z, operation):
    input_dtype = x1.get("dtype")
    shape_ori = x1.get("run_shape")
    x1_value = np.array(x1.get("value")).reshape(-1)
    tensor_len = len(x1_value)
    
    if isinstance(x2, dict):
        x2_value = np.array(x2.get("value")).reshape(-1)
    elif x2 is None:
        x2_value = 2.0 * np.ones((tensor_len,), input_dtype)
    else:
        x2_value = x2 * np.ones((tensor_len,), input_dtype)
    
    if isinstance(y1, dict):
        y1_value = np.array(y1.get("value")).reshape(-1)
    elif y1 is None:
        y1_value = x1_value
    else:
        y1_value = y1 * np.ones((tensor_len,), input_dtype)
    
    if isinstance(y2, dict):
        y2_value = np.array(y2.get("value")).reshape(-1)
    elif y2 is None:
        if isinstance(x2, dict):
            y2_value = x2_value
        else:
            y2_value = np.zeros((tensor_len,), input_dtype)
    else:
        y2_value = y2 * np.ones((tensor_len,), input_dtype)
    
    output = np.zeros((tensor_len,), input_dtype)
    if operation == "eq":
        for i in range(tensor_len):
            output[i] = y1_value[i] if x1_value[i] == x2_value[i] else y2_value[i]
    elif operation == "ge":
        for i in range(tensor_len):
            output[i] = y1_value[i] if x1_value[i] >= x2_value[i] else y2_value[i]
    elif operation == "gt":
        for i in range(tensor_len):
            output[i] = y1_value[i] if x1_value[i] > x2_value[i] else y2_value[i]
    elif operation == "le":
        for i in range(tensor_len):
            output[i] = y1_value[i] if x1_value[i] <= x2_value[i] else y2_value[i]
    elif operation == "lt":
        for i in range(tensor_len):
            output[i] = y1_value[i] if x1_value[i] < x2_value[i] else y2_value[i]
    else:
        for i in range(tensor_len):
            output[i] = y1_value[i] if x1_value[i] != x2_value[i] else y2_value[i]
    return output.reshape(shape_ori)

ut_case.add_precision_case(
    ["Ascend910A"], {
        "params": [
            {
                "shape": (-1, -1),
                "dtype": "float16",
                "range": [(1, 200), (1, 100)],
                "run_shape": (2, 64),
                "param_type": "input"
            },
            {
                "shape": (-1, -1),
                "dtype": "float16",
                "range": [(1, 200), (1, 100)],
                "run_shape": (2, 64),
                "param_type": "input"
            },
            {
                "shape": (-1, -1),
                "dtype": "float16",
                "range": [(1, 200), (1, 100)],
                "run_shape": (2, 64),
                "param_type": "input"
            },
            {
                "shape": (-1, -1),
                "dtype": "float16",
                "range": [(1, 200), (1, 100)],
                "run_shape": (2, 64),
                "param_type": "input"
            },
            {
                "shape": (-1, -1),
                "dtype": "float16",
                "range": [(1, 200), (1, 100)],
                "run_shape": (2, 64),
                "param_type": "output"
            },"eq"
        ],
        "calc_expect_func": calc_expect_func,
        "precision_standard": precision_info.PrecisionStandard(0.001, 0.001),
        "case_name": "test_dync_vcmpsel_prec_01"
    })

ut_case.add_precision_case(
    ["Ascend910A"], {
        "params": [
            {
                "shape": (-1, -1),
                "dtype": "float32",
                "range": [(1, 200), (1, 100)],
                "run_shape": (2, 64),
                "param_type": "input"
            },
            None,
            {
                "shape": (-1, -1),
                "dtype": "float32",
                "range": [(1, 200), (1, 100)],
                "run_shape": (2, 64),
                "param_type": "input"
            },
            {
                "shape": (-1, -1),
                "dtype": "float32",
                "range": [(1, 200), (1, 100)],
                "run_shape": (2, 64),
                "param_type": "input"
            },
            {
                "shape": (-1, -1),
                "dtype": "float32",
                "range": [(1, 200), (1, 100)],
                "run_shape": (2, 64),
                "param_type": "output"
            },"ne"
        ],
        "calc_expect_func": calc_expect_func,
        "precision_standard": precision_info.PrecisionStandard(0.0001, 0.0001),
        "case_name": "test_dync_vcmpsel_prec_02"
    })

ut_case.add_precision_case(
    ["Ascend910A"], {
        "params": [
            {
                "shape": (-1, -1),
                "dtype": "float16",
                "range": [(1, 200), (1, 100)],
                "run_shape": (2, 64),
                "param_type": "input"
            },
            {
                "shape": (-1, -1),
                "dtype": "float16",
                "range": [(1, 200), (1, 100)],
                "run_shape": (2, 64),
                "param_type": "input"
            },
            3.0,
            {
                "shape": (-1, -1),
                "dtype": "float16",
                "range": [(1, 200), (1, 100)],
                "run_shape": (2, 64),
                "param_type": "input"
            },
            {
                "shape": (-1, -1),
                "dtype": "float16",
                "range": [(1, 200), (1, 100)],
                "run_shape": (2, 64),
                "param_type": "output"
            }, "ge"
        ],
        "calc_expect_func": calc_expect_func,
        "precision_standard": precision_info.PrecisionStandard(0.001, 0.001),
        "case_name": "test_dync_vcmpsel_prec_03"
    })

ut_case.add_precision_case(
    ["Ascend910A"], {
        "params": [
            {
                "shape": (-1, -1),
                "dtype": "float32",
                "range": [(1, 200), (1, 100)],
                "run_shape": (32, 64),
                "param_type": "input"
            },
            2.0,
            {
                "shape": (-1, -1),
                "dtype": "float32",
                "range": [(1, 200), (1, 100)],
                "run_shape": (32, 64),
                "param_type": "input"
            },
            3.0,
            {
                "shape": (-1, -1),
                "dtype": "float32",
                "range": [(1, 200), (1, 100)],
                "run_shape": (32, 64),
                "param_type": "output"
            }, "lt"
        ],
        "calc_expect_func": calc_expect_func,
        "precision_standard": precision_info.PrecisionStandard(0.0001, 0.0001),
        "case_name": "test_dync_vcmpsel_prec_04"
    })
