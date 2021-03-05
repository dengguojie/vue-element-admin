# # -*- coding:utf-8 -*-
import numpy as np
from op_test_frame.ut import BroadcastOpUT
from op_test_frame.common import precision_info


ut_case = BroadcastOpUT("Addcmul")


def calc_expect_func(input_data, input_0, input_1, input_value, output_z):
    res = input_data["value"] + input_value["value"] * input_0["value"] * input_1["value"]
    return [res, ]


def generate_case(input_shape, input_0_shape, input_1_shape, input_value_shape, output_shape, dtype, dtype_value,
                  ori_format, format, range=None, precision_standard=None,
                  iscal=False, iserror=False, type_error=False, assert_error=False, calc_func=calc_expect_func):
    if range is None:
        range = [-5.0, 5.0]

    if dtype is "float16":
        input_value = np.random.uniform(range[0], range[1], input_shape).astype(np.float16)
        x0_value = np.random.uniform(range[0], range[1], input_0_shape).astype(np.float16)
        x1_value = np.random.uniform(range[0], range[1], input_1_shape).astype(np.float16)
    elif dtype is "float32":
        input_value = np.random.uniform(range[0], range[1], input_shape).astype(np.float32)
        x0_value = np.random.uniform(range[0], range[1], input_0_shape).astype(np.float32)
        x1_value = np.random.uniform(range[0], range[1], input_1_shape).astype(np.float32)
    else:
        raise RuntimeError("data_a type is not support".format(dtype))

    if dtype_value is "float16":
        v_value = np.random.uniform(range[0], range[1], input_value_shape).astype(np.float16)
    elif dtype_value is "float32":
        v_value = np.random.uniform(range[0], range[1], input_value_shape).astype(np.float32)
    elif dtype_value is "int32":
        v_value = np.random.uniform(range[0], range[1], input_value_shape).astype(np.int32)
    else:
        raise RuntimeError("data_x type is not support".format(dtype))

    case = {}
    addcmul_x = {
        "dtype": dtype,
        "ori_shape": input_shape,
        "shape": input_shape,
        "ori_format": ori_format,
        "format": format,
        "value": input_value,
        "param_type": "data_input"
    }
    addcmul_x0 = {
        "dtype": dtype,
        "ori_shape": input_0_shape,
        "shape": input_0_shape,
        "ori_format": ori_format,
        "format": format,
        "value": x0_value,
        "param_type": "data_x1"
    }
    addcmul_x1 = {
        "dtype": dtype,
        "ori_shape": input_1_shape,
        "shape": input_1_shape,
        "ori_format": ori_format,
        "format": format,
        "value": x1_value,
        "param_type": "data_x2"
    }
    addcmul_v = {
        "dtype": dtype_value,
        "ori_shape": input_value_shape,
        "shape": input_value_shape,
        "ori_format": "ND",
        "format": "ND",
        "value": v_value,
        "param_type": "value"
    }
    res = {
        "dtype": dtype,
        "ori_shape": output_shape,
        "shape": output_shape,
        "ori_format": "ND",
        "format": "ND",
        "param_type": "y"
    }

    case["params"] = [addcmul_x, addcmul_x0, addcmul_x1, addcmul_v, res]
    if iscal:
        case["calc_expect_func"] = calc_func
    if iserror:
        case["expect"] = RuntimeError
    if precision_standard is None:
        case["precision_standard"] = precision_info.PrecisionStandard(0.005, 0.005)
    else:
        case["precision_standard"] = precision_info.PrecisionStandard(precision_standard[0], precision_standard[1])

    return case


def test_op_select_format(test_arg):
    from impl.addcmul import op_select_format
    op_select_format({"shape": (1, 1), "dtype": "float16", "format": "ND", "ori_shape": (1, 1), "ori_format": "ND"},
                     {"shape": (1, 1), "dtype": "float16", "format": "ND", "ori_shape": (1, 1), "ori_format": "ND"},
                     {"shape": (1, 1), "dtype": "float16", "format": "ND", "ori_shape": (1, 1), "ori_format": "ND"},
                     {"shape": (1,), "dtype": "float16", "format": "ND", "ori_shape": (1,), "ori_format": "ND"},
                     {"shape": (1, 1), "dtype": "float16", "format": "ND", "ori_shape": (1, 1), "ori_format": "ND"})
    op_select_format({"shape": (20, 28, 3, 3, 16), "dtype": "float32", "format": "NC1HWC0", "ori_shape": (20, 28, 3, 3, 16), "ori_format": "NC1HWC0"},
                     {"shape": (20, 28, 3, 3, 16), "dtype": "float32", "format": "NC1HWC0", "ori_shape": (20, 28, 3, 3, 16), "ori_format": "NC1HWC0"},
                     {"shape": (20, 28, 3, 3, 16), "dtype": "float32", "format": "NC1HWC0", "ori_shape": (20, 28, 3, 3, 16), "ori_format": "NC1HWC0"},
                     {"shape": (1, ), "dtype": "float32", "format": "ND", "ori_shape": (1,), "ori_format": "ND"},
                     {"shape": (20, 28, 3, 3, 16), "dtype": "float32", "format": "NC1HWC0", "ori_shape": (20, 28, 3, 3, 16), "ori_format": "NC1HWC0"})
    op_select_format({"shape": (20, 28, 16, 16), "dtype": "float16", "format": "FRACTAL_NZ", "ori_shape": (20, 28, 16, 16), "ori_format": "FRACTAL_NZ"},
                     {"shape": (20, 28, 16, 16), "dtype": "float16", "format": "FRACTAL_NZ", "ori_shape": (20, 28, 16, 16), "ori_format": "FRACTAL_NZ"},
                     {"shape": (20, 28, 16, 16), "dtype": "float16", "format": "FRACTAL_NZ", "ori_shape": (20, 28, 16, 16), "ori_format": "FRACTAL_NZ"},
                     {"shape": (1, ), "dtype": "float16", "format": "ND", "ori_shape": (1,), "ori_format": "ND"},
                     {"shape": (20, 28, 16, 16), "dtype": "float16", "format": "FRACTAL_NZ", "ori_shape": (20, 28, 16, 16), "ori_format": "FRACTAL_NZ"})
    op_select_format({"shape": (20, 28, 16, 16), "dtype": "float16", "format": "FRACTAL_Z", "ori_shape": (20, 28, 16, 16), "ori_format": "FRACTAL_Z"},
                     {"shape": (20, 28, 16, 16), "dtype": "float16", "format": "FRACTAL_Z", "ori_shape": (20, 28, 16, 16), "ori_format": "FRACTAL_Z"},
                     {"shape": (20, 28, 16, 16), "dtype": "float16", "format": "FRACTAL_Z", "ori_shape": (20, 28, 16, 16), "ori_format": "FRACTAL_Z"},
                     {"shape": (1, ), "dtype": "float16", "format": "ND", "ori_shape": (1,), "ori_format": "ND"},
                     {"shape": (20, 28, 16, 16), "dtype": "float16", "format": "FRACTAL_Z", "ori_shape": (20, 28, 16, 16), "ori_format": "FRACTAL_Z"})


# UT test
# 1. test 2d input
ut_case.add_case(
    "all", case=generate_case((12, 12), (12, 12), (12, 12), (1,), (12, 12),
                              "float16", "float16", "ND", "ND", iserror=False))

# 2. test 4d input
ut_case.add_case(
    "all", case=generate_case((470, 3, 1, 1), (470, 3, 1, 1), (470, 3, 1, 1), (1,), (470, 3, 1, 1),
                              "float16", "float16", "ND", "ND", iserror=False))

ut_case.add_cust_test_func(test_func=test_op_select_format)
