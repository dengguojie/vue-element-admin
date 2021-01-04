#!/usr/bin/env python
# -*- coding: UTF-8 -*-
from op_test_frame.ut import OpUT

ut_case = OpUT("MaxPool", "impl.dynamic.max_pool", "max_pool")


def gen_dynamic_maxpool_case(shape_x, shape_y, ori_shape_x, ori_shape_y, range_x, range_y, format, ori_format, dtype_val, ksize, strides, padding, data_format, kernel_name_val, expect):
    return {"params": [{"shape": shape_x, "dtype": dtype_val, "ori_shape": ori_shape_x, "ori_format": ori_format, "format": format, "range": range_x},
                       {"shape": shape_y, "dtype": dtype_val, "ori_shape": ori_shape_y, "ori_format": ori_format, "format": format, "range": range_y},
                       ksize, strides, padding, data_format,],
            "case_name": kernel_name_val,
            "expect": expect,
            "format_expect": [],
            "support_expect": True}

ut_case.add_case("all",
                 gen_dynamic_maxpool_case((-1,-1,-1,-1,-1),(-1,-1,-1,-1,-1),(-1,-1,-1,-1),(-1,-1,-1,-1),
                                           ((1, None),(1, None),(1, None),(1, None),(1, None)),((1, None),(1, None),(1, None),(1, None),(1, None)),
                                           "NC1HWC0","NHWC","float16",[1,1,1,1],[1,1,1,1],"SAME","NHWC","max_pool_case", "success"))
ut_case.add_case("all",
                 gen_dynamic_maxpool_case((-1,-1,-1,-1,-1),(-1,-1,-1,-1,-1),(-1,-1,-1,-1),(-1,-1,-1,-1),
                                          ((1, None),(1, None),(1, None),(1, None),(1, None)),((1, None),(1, None),(1, None),(1, None),(1, None)),
                                          "NC1HWC0","NHWC","float16",[1,2,2,1],[1,2,2,1],"SAME","NHWC","max_pool_case", "success"))
ut_case.add_case("all",
                 gen_dynamic_maxpool_case((-1,-1,-1,-1,-1),(-1,-1,-1,-1,-1),(-1,-1,-1,-1),(-1,-1,-1,-1),
                                          ((1, None),(1, None),(1, None),(1, None),(1, None)),((1, None),(1, None),(1, None),(1, None),(1, None)),
                                          "NC1HWC0","NHWC","float16",[1,32,32,1],[1,32,32,1],"SAME","NHWC","max_pool_case", "success"))
ut_case.add_case("all",
                 gen_dynamic_maxpool_case((-1,-1,-1,-1,-1),(-1,-1,-1,-1,-1),(-1,-1,-1,-1),(-1,-1,-1,-1),
                                          ((1, None),(1, None),(1, None),(1, None),(1, None)),((1, None),(1, None),(1, None),(1, None),(1, None)),
                                          "NC1HWC0","NCHW","float16",[1,1,1,1],[1,1,1,1],"VALID","NCHW","max_pool_case", "success"))
ut_case.add_case("all",
                 gen_dynamic_maxpool_case((-1,-1,-1,-1,-1),(-1,-1,-1,-1,-1),(-1,-1,-1,-1),(-1,-1,-1,-1),
                                          ((1, None),(1, None),(1, None),(1, None),(1, None)),((1, None),(1, None),(1, None),(1, None),(1, None)),
                                          "NC1HWC0","NCHW","float16",[1,1,3,3],[1,1,3,3],"VALID","NCHW","max_pool_case", "success"))
ut_case.add_case("all",
                 gen_dynamic_maxpool_case((-1,-1,-1,-1,-1),(-1,-1,-1,-1,-1),(-1,-1,-1,-1),(-1,-1,-1,-1),
                                          ((1, None),(1, None),(1, None),(1, None),(1, None)),((1, None),(1, None),(1, None),(1, None),(1, None)),
                                          "NC1HWC0","NCHW","float16",[1,1,25,25],[1,1,25,25],"VALID","NCHW","max_pool_case", "success"))
ut_case.add_case("all",
                 gen_dynamic_maxpool_case((-1,-1,-1,-1,-1),(-1,-1,-1,-1,-1),(-1,-1,-1,-1),(-1,-1,-1,-1),
                                          ((1, None),(1, None),(1, None),(1, None),(1, None)),((1, None),(1, None),(1, None),(1, None),(1, None)),
                                          "ND","NHWC","float16",[1,1,1,1],[1,1,1,1],"SAME","NHWC","max_pool_case", RuntimeError))
ut_case.add_case("all",
                 gen_dynamic_maxpool_case((-1,-1,-1,-1),(-1,-1,-1,-1),(-1,-1,-1,-1),(-1,-1,-1,-1),
                                          ((1, None),(1, None),(1, None),(1, None),(1, None)),((1, None),(1, None),(1, None),(1, None),(1, None)),
                                          "NC1HWC0","NHWC","float16",[1,1,1,1],[1,1,1,1],"SAME","NHWC","max_pool_case", RuntimeError))
ut_case.add_case("all",
                 gen_dynamic_maxpool_case((-1,-1,-1,-1,-1),(-1,-1,-1,-1,-1),(-1,-1,-1,-1),(-1,-1,-1,-1),
                                          ((1, None),(1, None),(1, None),(1, None),(1, None)),((1, None),(1, None),(1, None),(1, None),(1, None)),
                                          "NC1HWC0","NHWC","float16",[1,1,1,1,1],[1,1,1,1],"SAME","NHWC","max_pool_case", RuntimeError))
ut_case.add_case("all",
                 gen_dynamic_maxpool_case((-1,-1,-1,-1,-1),(-1,-1,-1,-1,-1),(-1,-1,-1,-1),(-1,-1,-1,-1),
                                          ((1, None),(1, None),(1, None),(1, None),(1, None)),((1, None),(1, None),(1, None),(1, None),(1, None)),
                                          "NC1HWC0","NHWC","float16",[1,1,1,1],[1,1,1,1,1],"SAME","NHWC","max_pool_case", RuntimeError))
ut_case.add_case("all",
                 gen_dynamic_maxpool_case((-1,-1,-1,-1,-1),(-1,-1,-1,-1,-1),(-1,-1,-1,-1),(-1,-1,-1,-1),
                                          ((1, None),(1, None),(1, None),(1, None),(1, None)),((1, None),(1, None),(1, None),(1, None),(1, None)),
                                          "NC1HWC0","NHWC","float16",[1,1,1,1],[1,1,1,1],"SAME","ND","max_pool_case", RuntimeError))
ut_case.add_case("all",
                 gen_dynamic_maxpool_case((-1,-1,-1,-1,-1),(-1,-1,-1,-1,-1),(-1,-1,-1,-1),(-1,-1,-1,-1),
                                          ((1, None),(1, None),(1, None),(1, None),(1, None)),((1, None),(1, None),(1, None),(1, None),(1, None)),
                                          "NC1HWC0","NHWC","float16",[2,1,1,2],[1,1,1,1],"SAME","NHWC","max_pool_case", RuntimeError))
ut_case.add_case("all",
                 gen_dynamic_maxpool_case((-1,-1,-1,-1,-1),(-1,-1,-1,-1,-1),(-1,-1,-1,-1),(-1,-1,-1,-1),
                                          ((1, None),(1, None),(1, None),(1, None),(1, None)),((1, None),(1, None),(1, None),(1, None),(1, None)),
                                          "NC1HWC0","NHWC","float16",[1,1,1,1],[2,1,1,2],"SAME","NHWC","max_pool_case", RuntimeError))
ut_case.add_case("all",
                 gen_dynamic_maxpool_case((-1,-1,-1,-1,-1),(-1,-1,-1,-1,-1),(-1,-1,-1,-1),(-1,-1,-1,-1),
                                          ((1, None),(1, None),(1, None),(1, None),(1, None)),((1, None),(1, None),(1, None),(1, None),(1, None)),
                                          "NC1HWC0","NHWC","float16",[1,0,0,1],[1,1,1,1],"SAME","NHWC","max_pool_case", RuntimeError))
ut_case.add_case("all",
                 gen_dynamic_maxpool_case((-1,-1,-1,-1,-1),(-1,-1,-1,-1,-1),(-1,-1,-1,-1),(-1,-1,-1,-1),
                                          ((1, None),(1, None),(1, None),(1, None),(1, None)),((1, None),(1, None),(1, None),(1, None),(1, None)),
                                          "NC1HWC0","NHWC","float16",[1,1,1,1],[1,0,0,1],"SAME","NHWC","max_pool_case", RuntimeError))
ut_case.add_case("all",
                 gen_dynamic_maxpool_case((-1,-1,-1,-1,-1),(-1,-1,-1,-1,-1),(-1,-1,-1,-1),(-1,-1,-1,-1),
                                          ((1, None),(1, None),(1, None),(1, None),(1, None)),((1, None),(1, None),(1, None),(1, None),(1, None)),
                                          "NC1HWC0","NHWC","float16",[1,1,1,1],[1,1,1,1],"XXX","NHWC","max_pool_case", RuntimeError))

if __name__ == '__main__':
    import te
    with te.op.dynamic():
        ut_case.run("Ascend910")
    exit(0)
