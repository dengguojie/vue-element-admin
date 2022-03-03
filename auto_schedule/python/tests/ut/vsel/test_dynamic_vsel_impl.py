# # -*- coding:utf-8 -*-
from sch_test_frame.ut import OpUT
from sch_test_frame.utils.op_param_util import cartesian_set_format_dtype
from sch_test_frame.common import precision_info
import numpy as np

import tbe
from tbe import tvm
from tbe.common.utils import shape_util
from tbe.common.register import register_operator


@register_operator("vsel")
def dsl_dynamic_vsel(condition, x, y, z, kernel_name="dsl_dynamic_vsel"):
    condition_type = condition.get("dtype")
    schedules, tensors = [], []
    if isinstance(x, dict) and isinstance(y, dict):
        x_dtype = x.get("dtype")
        y_dtype = y.get("dtype")
        ins = tbe.dsl.classify([condition, x, y], "elewise")
        
        for (condition, x, y) in ins:
            with tbe.dsl.compute():
                shape_condition, shape_x, shape_y = shape_util.variable_shape([condition, x, y])
                data_1 = tvm.placeholder(shape_condition, name='data1', dtype=condition_type)
                data_2 = tvm.placeholder(shape_x, name='data2', dtype=x_dtype)
                data_3 = tvm.placeholder(shape_y, name='data3', dtype=y_dtype)
                res = tbe.dsl.vsel(data_1, data_2, data_3)

                tensors.append((data_1, data_2, data_3, res))

            with tvm.target.cce():
                sch = tbe.dsl.auto_schedule(res)
            schedules.append(sch)
    elif isinstance(x, dict) and not isinstance(y, dict):
        x_dtype = x.get("dtype")
        ins = tbe.dsl.classify([condition, x], "elewise")
        
        for (condition, x) in ins:
            with tbe.dsl.compute():
                shape_condition, shape_x = shape_util.variable_shape([condition, x])
                data_1 = tvm.placeholder(shape_condition, name='data1', dtype=condition_type)
                data_2 = tvm.placeholder(shape_x, name='data2', dtype=x_dtype)
                res = tbe.dsl.vsel(data_1, data_2, y)

                tensors.append((data_1, data_2, res))

            with tvm.target.cce():
                sch = tbe.dsl.auto_schedule(res)
            schedules.append(sch)
    elif isinstance(y, dict) and not isinstance(x, dict):
        y_dtype = y.get("dtype")
        ins = tbe.dsl.classify([condition, y], "elewise")
        
        for (condition, y) in ins:
            with tbe.dsl.compute():
                shape_condition, shape_y = shape_util.variable_shape([condition, y])
                data_1 = tvm.placeholder(shape_condition, name='data1', dtype=condition_type)
                data_2 = tvm.placeholder(shape_y, name='data2', dtype=y_dtype)
                res = tbe.dsl.vsel(data_1, x, data_2)

                tensors.append((data_1, data_2, res))

            with tvm.target.cce():
                sch = tbe.dsl.auto_schedule(res)
            schedules.append(sch)
    else:
        ins = tbe.dsl.classify([condition], "elewise")
        
        for (condition,) in ins:
            with tbe.dsl.compute():
                shape_condition = shape_util.variable_shape([condition])[0]
                data_1 = tvm.placeholder(shape_condition, name='data1', dtype=condition_type)
                res = tbe.dsl.vsel(data_1, x, y)

                tensors.append((data_1, res))

            with tvm.target.cce():
                sch = tbe.dsl.auto_schedule(res)
            schedules.append(sch)

    config = {"name": kernel_name, "tensor_list": tensors}
    tbe.dsl.build(schedules, config)


ut_case = OpUT("vsel", "vsel.test_dynamic_vsel_impl", "dsl_dynamic_vsel")

case1 = {
    "params": [{
        "shape": (-1, -1),
        "dtype": "bool",
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
    }],
    "case_name":
        "test_dync_vsel_1",
    "expect":
        "success",
    "support_expect":
        True
}

case2 = {
    "params": [{
        "shape": (-1, -1),
        "dtype": "bool",
        "range": [(1, None), (1, None)]
    },
    0.5, 
    {
        "shape": (-1, -1),
        "dtype": "float16",
        "range": [(1, None), (1, None)]
    }, {
        "shape": (-1, -1),
        "dtype": "float16",
        "range": [(1, None), (1, None)]
    }],
    "case_name":
        "test_dync_vsel_2",
    "expect":
        "success",
    "support_expect":
        True
}


ut_case.add_case(["Ascend910A", "Ascend710"], case1)
ut_case.add_case(["Ascend910A", "Ascend710"], case2)


def calc_expect_func(condition, x, y, z):
    condition_dtype = condition.get("dtype")
    condition_run_shape = condition.get("run_shape")
    condition_value = np.array(condition.get("value")).reshape(-1)
    condition_size = len(condition_value)
    if condition_dtype == "bool":
        if isinstance(x, dict):
            x_value = np.array(x.get("value")).reshape(-1)
        else:
            dtype = y.get("dtype") if isinstance(y, dict) else np.float16
            x_value = np.zeros((condition_size,), dtype)
        if isinstance(y, dict):
            y_value = np.array(y.get("value")).reshape(-1)
        else:
            dtype = x.get("dtype") if isinstance(x, dict) else np.float16
            y_value = np.zeros((condition_size,), dtype)
        output = []
        for i in range(condition_size):
            if condition_value[i] == 1:
                output.append(x_value[i])
            else:
                output.append(y_value[i])
        return np.array(output).reshape(condition_run_shape)
    else:
        if isinstance(x, dict):
            x_value = np.array(x.get("value")).reshape(-1)
        else:
            x_value = np.zeros((condition_size,), np.float16)
        if isinstance(y, dict):
            y_value = np.array(y.get("value")).reshape(-1)
        else:
            y_value = np.zeros((condition_size,), np.float16)
        output = []
        for i in range(condition_size):
            mask = condition_value[i]
            for j in range(8):
                flag = mask % 2
                mask = mask // 2
                real_index = i * 8 + j
                if flag == 1:
                    output[real_index].append(x_value[real_index])
                else:
                    output[real_index].append(y_value[real_index])
        return np.array(output).reshape(condition_run_shape)

ut_case.add_precision_case(
    ["Ascend910A"], {
        "params": [
            {
                "shape": (-1, -1),
                "dtype": "bool",
                "range": [(1, 200), (1, 100)],
                "run_shape": (2, 80),
                "param_type": "input"
            },
            {
                "shape": (-1, -1),
                "dtype": "float16",
                "range": [(1, 200), (1, 100)],
                "run_shape": (2, 80),
                "param_type": "input"
            },
            {
                "shape": (-1, -1),
                "dtype": "float16",
                "range": [(1, 200), (1, 100)],
                "run_shape": (2, 80),
                "param_type": "input"
            },
            {
                "shape": (-1, -1),
                "dtype": "float16",
                "range": [(1, 200), (1, 100)],
                "run_shape": (2, 80),
                "param_type": "output"
            }
        ],
        "calc_expect_func": calc_expect_func,
        "precision_standard": precision_info.PrecisionStandard(0.001, 0.001),
        "case_name": "test_dync_vsel_prec_01"
    })
