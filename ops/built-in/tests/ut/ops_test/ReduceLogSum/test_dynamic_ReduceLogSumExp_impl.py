#!/usr/bin/env python
# -*- coding: UTF-8 -*-
from op_test_frame.ut import OpUT

ut_case = OpUT("ReduceLogSum", "impl.dynamic.reduce_log_sum", "reduce_log_sum")

def gen_dynamic_reduce_log_sum_case(shape_x, range_x, 
                                        shape_axes, range_axes, 
                                        shape_y, range_y, 
                                        dtype_val, data_format, 
                                        ori_shape_x, ori_shape_axes, ori_shape_y, 
                                        keepdims, 
                                        kernel_name_val, 
                                        expect):

    return {"params": [{"shape": shape_x, "dtype": dtype_val,
                        "range": range_x, "format": data_format,
                        "ori_shape": ori_shape_x, "ori_format": data_format},
                       {"shape": shape_axes, "dtype": "int32",
                        "range": range_axes, "format": data_format,
                        "ori_shape": ori_shape_axes, "ori_format": data_format},
                       {"shape": shape_y, "dtype": dtype_val,
                        "range": range_y, "format": data_format,
                        "ori_shape": ori_shape_y, "ori_format": data_format}, keepdims],
            "case_name": kernel_name_val,
            "expect": expect,
            "format_expect": [],
            "support_expect": True}

ut_case.add_case("all", gen_dynamic_reduce_log_sum_case((-1,3,-1,2), [(1,None), (3,3), (1,None), (2,2)],
                                       (1,), [(1,1),], 
                                       (-1,1,-1,2), [(1,None), (1,1), (1,None), (2,2)], 
                                       "float16", "ND", 
                                       (4,3,4,2), (1, ), (4,1,4,2),
                                       True, 
                                       "dynamic_reduce_log_sum_float16_true", 
                                       "success"))
ut_case.add_case("all", gen_dynamic_reduce_log_sum_case((4,3,4,2), [(4,4), (3,3), (4,4), (2,2)],
                                       (1,), [(1,1),], 
                                       (4,4,2), [(4,4), (4,4), (2,2)], 
                                       "float32", "ND", 
                                       (4,3,4,2), (1, ), (4,4,2),
                                       False, 
                                       "static_reduce_log_sum_float32_false", 
                                       "success"))

if __name__ == "__main__":
    ut_case.run("Ascend910A")
    ut_case.run("Ascend310")

