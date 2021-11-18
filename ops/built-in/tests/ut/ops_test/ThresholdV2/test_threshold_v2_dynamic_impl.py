# -*- coding: UTF-8 -*-
from op_test_frame.ut import OpUT

ut_case = OpUT("ThresholdV2", "impl.dynamic.threshold_v2", "threshold_v2")

def gen_use_value_case(shape_val, dtype_val, range_val):
    return{
        "params": [
            {"ori_shape": shape_val, "shape": shape_val,
            "ori_format": "ND", "format": "ND",
            "dtype": dtype_val, "range": range_val},

            {"ori_shape": (1,), "shape": (1,),
            "ori_format": "ND", "format": "ND",
            "dtype": dtype_val},

            {"ori_shape": (1,), "shape": (1,),
            "ori_format": "ND", "format": "ND",
            "dtype": dtype_val},

            {"ori_shape": shape_val, "shape": shape_val,
            "ori_format": "ND", "format": "ND",
            "dtype": dtype_val, "range": range_val}],
        "case_name": "use_value_" + dtype_val
    }

def gen_no_value_case(shape_val, dtype_val, range_val):
    return{
        "params": [
            {"ori_shape": shape_val, "shape": shape_val,
            "ori_format": "ND", "format": "ND",
            "dtype": dtype_val, "range": range_val},

            {"ori_shape": (1,), "shape": (1,),
            "ori_format": "ND", "format": "ND",
            "dtype": dtype_val},

            None,

            {"ori_shape": shape_val, "shape": shape_val,
            "ori_format": "ND", "format": "ND",
            "dtype": dtype_val, "range": range_val}],
        "case_name": "no_value_" + dtype_val
    }

ut_case.add_case("all", {
    "params": [
        {"ori_shape": (-1,), "shape": (-1,), "ori_format": "ND", "format": "ND", "dtype": "float16", "range": ((1, None),)},
        {"ori_shape": (2, 2), "shape": (2, 2), "ori_format": "ND", "format": "ND", "dtype": "float16"},
        {"ori_shape": (2, 2), "shape": (2, 2), "ori_format": "ND", "format": "ND", "dtype": "float16"},
        {"ori_shape": (-1,), "shape": (-1,), "ori_format": "ND", "format": "ND", "dtype": "float16", "range": ((1, None),)}],
    "case_name": "para_check",
    "expect": RuntimeError
})

ut_case.add_case("Ascend910A", gen_use_value_case((-1,), "float32", ((1, None),)))
ut_case.add_case("Ascend910A", gen_use_value_case((-1, -1, -1), "float32", ((1, None), (1, None), (1, None))))
ut_case.add_case("all", gen_use_value_case((-1,), "float16", ((1, None),)))
ut_case.add_case("all", gen_use_value_case((-1,), "int32", ((1, None),)))
ut_case.add_case("all", gen_use_value_case((-1,), "uint8", ((1, None),)))
ut_case.add_case("all", gen_use_value_case((-1,), "int8", ((1, None),)))

ut_case.add_case("Ascend910A", gen_no_value_case((-1,), "float32", ((1, None),)))
ut_case.add_case("Ascend910A", gen_no_value_case((-1, -1, -1), "float32", ((1, None), (1, None), (1, None))))
ut_case.add_case("all", gen_no_value_case((-1,), "float16", ((1, None),)))
ut_case.add_case("all", gen_no_value_case((-1,), "int32", ((1, None),)))
ut_case.add_case("all", gen_no_value_case((-1,), "uint8", ((1, None),)))
ut_case.add_case("all", gen_no_value_case((-1,), "int8", ((1, None),)))

if __name__ == "__main__":
    ut_case.run("Ascend910A")
    #ut_case.run("Ascend310")
