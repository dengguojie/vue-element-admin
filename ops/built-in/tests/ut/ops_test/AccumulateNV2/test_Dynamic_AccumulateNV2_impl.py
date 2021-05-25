#!/usr/bin/env python
# -*- coding: UTF-8 -*-
#  pylint: disable=invalid-name,missing-module-docstring
from op_test_frame.ut import OpUT

ut_case = OpUT("AccumulateNV2", "impl.dynamic.accumulate_nv2", "accumulate_nv2")

# pylint: disable=too-many-arguments
def gen_dynamic_accumulate_nv2_case(shape_x, range_x, dtype_val, tensor_num,
                           kernel_name_val, expect):
    """
    gen_params
    """
    return {"params": [[{"ori_shape": shape_x, "shape": shape_x, "ori_format": "ND",
                         "format": "ND", "dtype": dtype_val, "range": range_x},
                        {"ori_shape": shape_x, "shape": shape_x, "ori_format": "ND",
                         "format": "ND", "dtype": dtype_val, "range": range_x}],
                       {"ori_shape": shape_x, "shape": shape_x, "ori_format": "ND",
                        "format": "ND", "dtype": dtype_val, "range": range_x},
                       tensor_num],
        "case_name": kernel_name_val,
        "expect": expect,
        "format_expect": [],
        "support_expect": True}


ut_case.add_case("all",
                 gen_dynamic_accumulate_nv2_case((-1,), ((1, None),), "float16", 2,
                                        "dynamic_add_n_fp16_ND", "success"))
ut_case.add_case("all",
                 gen_dynamic_accumulate_nv2_case((-1,), ((1, None),), "float32", 3,
                                        "dynamic_add_n_fp32_f_ND", "failed"))
ut_case.add_case("all",
                 gen_dynamic_accumulate_nv2_case((-1,), ((1, None),), "float32", 2,
                                        "dynamic_add_n_fp32_ND", "success"))
if __name__ == '__main__':
    ut_case.run("Ascend910A")
    ut_case.run("Ascend310")
