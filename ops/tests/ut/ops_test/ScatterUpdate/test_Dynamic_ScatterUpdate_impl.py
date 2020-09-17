#!/usr/bin/env python
# -*- coding: UTF-8 -*-
from op_test_frame.ut import OpUT

ut_case = OpUT("ScatterUpdate", "impl.dynamic.scatter_update", "scatter_update")

# TODO fix me, run failed
# def gen_dynamic_floormod_case(shape_x, shape_y, range_x, range_y, dtype_val, kernel_name_val, expect):
#     return {"params": [{"shape": shape_x, "dtype": dtype_val, "range": range_x},
#                        {"shape": shape_y, "dtype": dtype_val, "range": range_y},
#                        {"shape": shape_y, "dtype": dtype_val, "range": range_y}],
#             "case_name": kernel_name_val,
#             "expect": expect,
#             "format_expect": [],
#             "support_expect": True}
#
# ut_case.add_case("all",
#                  gen_dynamic_floormod_case((-1,), (1,),
#                                            ((1, None),), ((1, 1),),
#                                            "float16", "dynamic_floormod_fp16_ND", "success"))


if __name__ == '__main__':
    ut_case.run()
    exit(0)
