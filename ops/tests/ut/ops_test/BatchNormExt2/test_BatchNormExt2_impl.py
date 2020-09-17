#!/usr/bin/env python
# -*- coding: UTF-8 -*-
from op_test_frame.ut import OpUT

ut_case = OpUT("BatchNormExt2", None, None)


def gen_batch_norm_ext2_case(shape_x, shape_scale, shape_offset, shape_mean, shape_variance, dtype_x,
                             dtype_other, format, case_name_val, expect):
    return {
        "params": [{"shape": shape_x, "dtype": dtype_x, "ori_shape": shape_x, "ori_format": format, "format": format},
                   {"shape": shape_scale, "dtype": dtype_other, "ori_shape": shape_scale, "ori_format": format,
                    "format": format},
                   {"shape": shape_offset, "dtype": dtype_other, "ori_shape": shape_offset, "ori_format": format,
                    "format": format},
                   None,
                   None,
                   {"shape": shape_x, "dtype": dtype_x, "ori_shape": shape_x, "ori_format": format, "format": format},
                   {"shape": shape_scale, "dtype": dtype_other, "ori_shape": shape_scale, "ori_format": format,
                    "format": format},
                   {"shape": shape_offset, "dtype": dtype_other, "ori_shape": shape_offset, "ori_format": format,
                    "format": format},
                   {"shape": shape_mean, "dtype": dtype_other, "ori_shape": shape_mean, "ori_format": format,
                    "format": format},
                   {"shape": shape_variance, "dtype": dtype_other, "ori_shape": shape_variance, "ori_format": format,
                    "format": format}],
        "case_name": case_name_val,
        "expect": expect,
        "format_expect": [],
        "support_expect": True}


ut_case.add_case(["Ascend310", "Ascend710", "Ascend910"],
                 gen_batch_norm_ext2_case((10, 20, 30, 40), (40,), (40,), (40,), (40,), "float32",
                                          "float32", "NHWC", "batch_norm_ext2_1", "success"))

if __name__ == '__main__':
    ut_case.run("Ascend910")
