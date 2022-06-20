#!/usr/bin/env python
# -*- coding: UTF-8 -*-
from op_test_frame.ut import OpUT

ut_case = OpUT("BatchNorm", "impl.batch_norm", "get_op_support_info")

def gen_batch_norm_case(shape_x, shape_scale, shape_mean, shape_reserve, dtype_x,
                        dtype_other, format, case_name_val, expect):
    return {"params": [{"shape": shape_x, "dtype": dtype_x, "ori_shape": shape_x, "ori_format": format, "format": format},
                       {"shape": shape_scale, "dtype": dtype_other, "ori_shape": shape_scale, "ori_format": format, "format": format},
                       {"shape": shape_scale, "dtype": dtype_other, "ori_shape": shape_scale, "ori_format": format, "format": format},
                       None,
                       None,
                       {"shape": shape_x, "dtype": dtype_x, "ori_shape": shape_x, "ori_format": format, "format": format},
                       {"shape": shape_scale, "dtype": dtype_other, "ori_shape": shape_scale, "ori_format": format, "format": format},
                       {"shape": shape_scale, "dtype": dtype_other, "ori_shape": shape_scale, "ori_format": format, "format": format},
                       {"shape": shape_reserve, "dtype": dtype_other, "ori_shape": shape_reserve, "ori_format": format, "format": format},
                       {"shape": shape_reserve, "dtype": dtype_other, "ori_shape": shape_reserve, "ori_format": format, "format": format}],
            "case_name": case_name_val,
            "expect": expect,
            "format_expect": [],
            "support_expect": True}


ut_case.add_case(["Ascend310", "Ascend310P3", "Ascend910"],
                 gen_batch_norm_case((1,2,3,4,16), (1,2,1,1,16), (0,), (), "float16",
                                     "float32", "NC1HWC0", "batch_norm_1", "success"))
ut_case.add_case(["Ascend310", "Ascend310P3", "Ascend910"],
                 gen_batch_norm_case((1,2,3,4,16), (2,2,1,1,16), (0,), (), "float16",
                                     "float32", "NC1HWC0", "batch_norm_2", RuntimeError))
ut_case.add_case(["Ascend310", "Ascend310P3", "Ascend910"],
                 gen_batch_norm_case((2,16,384,576,16), (1,16,1,1,16), (1,16,1,1,16), (), "float16",
                                     "float32", "NC1HWC0", "batch_norm_3", "success"))

def gen_batch_norm_case2(shape_x, shape_scale, shape_mean, shape_reserve, dtype_x,
                         dtype_other, format, ori_format, is_train, is_reverse, case_name_val, expect):
    format = ori_format
    return {"params": [{"shape": shape_x, "dtype": dtype_x, "ori_shape": shape_x, "ori_format": format, "format": format},
                       {"shape": shape_scale, "dtype": dtype_other, "ori_shape": shape_scale, "ori_format": format, "format": format},
                       {"shape": shape_scale, "dtype": dtype_other, "ori_shape": shape_scale, "ori_format": format, "format": format},
                       None if not is_reverse else {"shape": shape_scale, "dtype": dtype_other, "ori_shape": shape_scale, "ori_format": format, "format": format},
                       None if not is_reverse else {"shape": shape_scale, "dtype": dtype_other, "ori_shape": shape_scale, "ori_format": format, "format": format},
                       {"shape": shape_x, "dtype": dtype_x, "ori_shape": shape_x, "ori_format": format, "format": format},
                       {"shape": shape_scale, "dtype": dtype_other, "ori_shape": shape_scale, "ori_format": format, "format": format},
                       {"shape": shape_scale, "dtype": dtype_other, "ori_shape": shape_scale, "ori_format": format, "format": format},
                       {"shape": shape_reserve, "dtype": dtype_other, "ori_shape": shape_reserve, "ori_format": format, "format": format},
                       {"shape": shape_reserve, "dtype": dtype_other, "ori_shape": shape_reserve, "ori_format": format, "format": format},
                       None,
                       0.001, ori_format, is_train],
            "case_name": case_name_val,
            "expect": expect,
            "format_expect": [],
            "support_expect": True}

ut_case.add_case(["Ascend310", "Ascend310P3", "Ascend910"],
                 gen_batch_norm_case2((2,16,384,576,16), (1,16,1,1,16), (1,16,1,1,16), (), "float16",
                                     "float32", "NC1HWC0", "NHWC",  False, True, "batch_norm_3", "success"))
ut_case.add_case(["Ascend310", "Ascend310P3", "Ascend910"],
                 gen_batch_norm_case2((2,16,384,576,16), (1,16,1,1,16), (1,16,1,1,16), (), "float16",
                                     "float32", "NC1HWC0", "NHWC",  True, False, "batch_norm_3", "success"))
ut_case.add_case(["Ascend310", "Ascend310P3", "Ascend910"],
                 gen_batch_norm_case2((2,16,384,576,16), (1,16,1,1,16), (1,16,1,1,16), (), "float16",
                                     "float32", "NC1HWC0", "NHWC",  False, False, "batch_norm_3", "success"))
ut_case.add_case(["Ascend310", "Ascend310P3", "Ascend910"],
                 gen_batch_norm_case2((2,16,384,576,16), (1,16,1,1,16), (1,16,1,1,16), (), "float16",
                                     "float32", "NC1HWC0", "NHWC",  True, True, "batch_norm_3", "success"))
ut_case.add_case(["Ascend310", "Ascend310P3", "Ascend910"],
                 gen_batch_norm_case2((2,16,384,576,16), (1,16,1,1,16), (1,16,1,1,16), (), "float16",
                                     "float32", "NC1HWC0", "NCHW",  False, True, "batch_norm_3", "success"))
ut_case.add_case(["Ascend310", "Ascend310P3", "Ascend910"],
                 gen_batch_norm_case2((2,16,384,576,16), (1,16,1,1,16), (1,16,1,1,16), (), "float16",
                                     "float32", "NC1HWC0", "NCHW",  True, False, "batch_norm_3", "success"))
ut_case.add_case(["Ascend310", "Ascend310P3", "Ascend910"],
                 gen_batch_norm_case2((2,16,384,576,16), (1,16,1,1,16), (1,16,1,1,16), (), "float16",
                                     "float32", "NC1HWC0", "NCHW",  False, False, "batch_norm_3", "success"))
ut_case.add_case(["Ascend310", "Ascend310P3", "Ascend910"],
                 gen_batch_norm_case2((2,16,384,576,16), (1,16,1,1,16), (1,16,1,1,16), (), "float16",
                                     "float32", "NC1HWC0", "NCHW",  True, True, "batch_norm_3", "success"))

ut_case.add_case(["Ascend310", "Ascend310P3", "Ascend910"],
                 gen_batch_norm_case2((2,16,384,576,16), (1,16,1,1,16), (1,16,1,1,16), (), "float16",
                                     "float32", "NC1HWC0", "NC1HWC0",  False, True, "batch_norm_3", "success"))
ut_case.add_case(["Ascend310", "Ascend310P3", "Ascend910"],
                 gen_batch_norm_case2((2,16,384,576,16), (1,16,1,1,16), (1,16,1,1,16), (), "float16",
                                     "float32", "NC1HWC0", "NC1HWC0",  False, False, "batch_norm_3", "success"))
ut_case.add_case(["Ascend310", "Ascend310P3", "Ascend910"],
                 gen_batch_norm_case2((2,16,384,576,16), (1,16,1,1,16), (1,16,1,1,16), (), "float16",
                                     "float32", "NC1HWC0", "NC1HWC0",  True, True, "batch_norm_3", "success"))
ut_case.add_case(["Ascend310", "Ascend310P3", "Ascend910"],
                 gen_batch_norm_case2((2,16,384,576,16), (1,16,1,1,16), (1,16,1,1,16), (), "float16",
                                     "float32", "NC1HWC0", "NC1HWC0",  True, False, "batch_norm_3", "success"))
ut_case.add_case(["Ascend310", "Ascend310P3", "Ascend910"],
                 gen_batch_norm_case2((2,16,384,576,16), (1,16,1,1,16), (1,16,1,1,16), (), "float16",
                                     "float32", "NC1HWC0", "NC1HWC1",  True, False, "batch_norm_3", "success"))
if __name__ == '__main__':
    ut_case.run("Ascend910")
    exit(0)
