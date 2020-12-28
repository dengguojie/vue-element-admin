import sys
from op_test_frame.ut import BroadcastOpUT
from op_test_frame.common import precision_info

ut_case = BroadcastOpUT("dropout_v2")


def generate_case(input0_shape, input1_shape, out_shape, prob, dtype, dtype2, range=None, precision_standard=None,
                  iscal=False, iserror=False):
    if range is None:
        range = [-255.0, 255.0]
    case = {}
    input0 = {
        "dtype": dtype,
        "ori_shape": input0_shape,
        "shape": input0_shape,
        "ori_format": "ND",
        "format": "ND",
        "range": range,
        "param_type": "input"
    }
    input1 = {
        "dtype": dtype2,
        "ori_shape": input1_shape,
        "shape": input1_shape,
        "ori_format": "ND",
        "format": "ND",
        "range": range,
        "param_type": "input"
    }
    out = {
        "dtype": dtype,
        "ori_shape": out_shape,
        "shape": out_shape,
        "ori_format": "ND",
        "format": "ND",
        "param_type": "output"
    }
    mask = {
        "dtype": dtype2,
        "ori_shape": out_shape,
        "shape": out_shape,
        "ori_format": "ND",
        "format": "ND",
        "param_type": "output"
    }
    new_seed = {
        "dtype": dtype2,
        "ori_shape": input1_shape,
        "shape": input1_shape,
        "ori_format": "ND",
        "format": "ND",
        "param_type": "output"
    }
    case["params"] = [input0, input1, out, mask, new_seed, prob]
    if iserror:
        case["expect"] = RuntimeError
    if precision_standard is None:
        case["precision_standard"] = precision_info.PrecisionStandard(0.005, 0.005)
    else:
        case["precision_standard"] = precision_info.PrecisionStandard(precision_standard[0], precision_standard[1])
    return case


ut_case.add_case("all",
                 case=generate_case((16, 132, 1024), (32 * 1024 * 12,), (16, 132, 1024), 0.3, "float32", "float32"))
ut_case.add_case("all", case=generate_case((16, 1024), (32 * 1024 * 12,), (16, 1024), 0.3, "float16", "float32"))
ut_case.add_case("all", case=generate_case((393220,), (32 * 1024 * 12,), (393220,), 0.3, "float16", "float32"))

ut_case.add_case("all", case=generate_case((16, 1024), (32 * 1024 * 12,), (16, 1024, 2), 0.3, "float16", "float32",
                                           iserror=True))
ut_case.add_case("all",
                 case=generate_case((16, 1024), (32 * 1024 * 12,), (16, 1024), 0.3, "int8", "float32", iserror=True))
ut_case.add_case("all",
                 case=generate_case((16, 1024), (32 * 1024 * 12,), (16, 1024), 0.3, "float16", "int8", iserror=True))
