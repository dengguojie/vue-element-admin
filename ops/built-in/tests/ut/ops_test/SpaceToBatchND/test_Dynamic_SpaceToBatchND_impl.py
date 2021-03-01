#!/usr/bin/env python
# -*- coding: UTF-8 -*-
from op_test_frame.ut import OpUT

ut_case = OpUT("SpaceToBatchND", "impl.dynamic.space_to_batch_nd", "space_to_batch_nd")


def gen_dynamic_spacetobatchnd_case(shape_x, shape_y, ori_shape_x, ori_shape_y, range_x, range_y, in_format, ori_format, dtype_val, kernel_name_val, shape_block, shape_pads, expect):
    return {"params": [{"shape": shape_x, "dtype": dtype_val, "ori_shape": ori_shape_x, "ori_format": ori_format, "format": in_format, "range": range_x},
                       {"shape": shape_block, "dtype": "int32", "ori_shape": shape_block, "ori_format": "ND", "format": "ND", "range": ((1, None),)},
                       {"shape": shape_pads, "dtype": "int32", "ori_shape": shape_pads, "ori_format": "ND", "format": "ND", "range": ((1, None),(1, None))},
                       {"shape": shape_y, "dtype": dtype_val, "ori_shape": ori_shape_y, "ori_format": ori_format, "format": in_format, "range": range_y},],
            "case_name": kernel_name_val,
            "expect": expect,
            "format_expect": [],
            "support_expect": True}

ut_case.add_case("all",
                 gen_dynamic_spacetobatchnd_case((-1,-1,-1,-1,-1),(-1,-1,-1,-1,-1),(-1,-1,-1,-1),(-1,-1,-1,-1),
                                           ((1, None),(1, None),(1, None),(1, None),(1, None)),((1, None),(1, None),(1, None),(1, None),(1, None)),
                                           "NC1HWC0","NHWC","float16","batchtospace_case",(2,),(2,2),"success"))
ut_case.add_case("all",
                 gen_dynamic_spacetobatchnd_case((-1,-1,-1,-1,-1,-1),(-1,-1,-1,-1,-1,-1),(-1,-1,-1,-1,-1),(-1,-1,-1,-1,-1),
                                                 ((1, None),(1, None),(1, None),(1, None),(1, None),(1, None)),((1, None),(1, None),(1, None),(1, None),(1, None),(1, None)),
                                                 "NDC1HWC0","NDHWC","float32","batchtospace_case",(3,),(3,2),"success"))
ut_case.add_case("all",
                 gen_dynamic_spacetobatchnd_case((-1,-1,-1,-1,-1),(-1,-1,-1,-1,-1),(-1,-1,-1,-1),(-1,-1,-1,-1),
                                               ((1, None),(1, None),(1, None),(1, None),(1, None)),((1, None),(1, None),(1, None),(1, None),(1, None)),
                                               "ND","NHWC","float16","batchtospace_case",(2,),(2,2),RuntimeError))
ut_case.add_case("all",
                 gen_dynamic_spacetobatchnd_case((-2,),(-2,),(-2,),(-2,),
                                           ((1, None),),((1, None),),
                                           "NC1HWC0","NHWC","float16","batchtospace_case",(2,),(2,2),"success"))

if __name__ == '__main__':
    import tbe
    with tbe.common.context.op_context.OpContext("dynamic"):
        ut_case.run("Ascend910A")
    exit(0)
