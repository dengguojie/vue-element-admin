#!/usr/bin/env python
# -*- coding: UTF-8 -*-
from op_test_frame.ut import OpUT

ut_case = OpUT("AscendQuant", "impl.dynamic.ascend_quant", "ascend_quant")

def gen_ascend_quant_case(shape_x, shape_y, ori_shape_x, ori_shape_y, range_x, range_y, format, ori_format, dtype_val, scale, offset, sqrt_mode, round_mode, kernel_name_val, expect):
    return {"params": [{"shape": shape_x, "dtype": dtype_val, "ori_shape": ori_shape_x, "ori_format": ori_format, "format": format, "range": range_x},
                       {"shape": shape_y, "dtype": "int8", "ori_shape": ori_shape_y, "ori_format": ori_format, "format": format, "range": range_y},
                       scale, offset, sqrt_mode, round_mode],
            "case_name": kernel_name_val,
            "expect": expect,
            "format_expect": [],
            "support_expect": True}

ut_case.add_case("Ascend910A",
                 gen_ascend_quant_case((-1,-1,-1,-1,16),(-1,-1,-1,-1,32),(-1,-1,-1,-1),(-1,-1,-1,-1),
                                      ((1, None),(1, None),(1, None),(1, None),(16, 16)),((1, None),(1, None),(1, None),(1, None),(32, 32)),
                                      "NC1HWC0","NHWC","float16",1.0,-3.0,False,"Round","ascend_quant_case_1", "success"))
ut_case.add_case("Ascend910A",
                 gen_ascend_quant_case((-1,-1,-1,-1,16),(-1,-1,-1,-1,32),(-1,-1,-1,-1),(-1,-1,-1,-1),
                                      ((1, None),(1, None),(1, None),(1, None),(16, 16)),((1, None),(1, None),(1, None),(1, None),(32, 32)),
                                      "NC1HWC0","NHWC","float32",2.0,0.0,True,"Round","ascend_quant_case_2", "success"))
ut_case.add_case("Ascend910A",
                 gen_ascend_quant_case((-1,-1,-1,-1,16),(-1,-1,-1,-1,32),(-1,-1,-1,-1),(-1,-1,-1,-1),
                                      ((1, None),(1, None),(1, None),(1, None),(16, 16)),((1, None),(1, None),(1, None),(1, None),(32, 32)),
                                      "NC1HWC0","NHWC","float16",3.0,-4.0,True,"Round","ascend_quant_case_3", "success"))

ut_case.add_case("Ascend910A",
                 gen_ascend_quant_case((-1,-1,16,16),(-1,-1,16,32),(-1,-1),(-1,-1),
                                      ((1, None),(1, None),(16, 16),(16, 16)),((1, None),(1, None),(16, 16),(32, 32)),
                                      "FRACTAL_NZ","ND","float16",5.5,-4.3,True,"Round","ascend_quant_case_4", "success"))
ut_case.add_case("Ascend910A",
                 gen_ascend_quant_case((-1,-1,-1,16,16),(-1,-1,-1,16,32),(-1,-1,-1),(-1,-1,-1),
                                      ((1, None),(1, None),(1, None),(16, 16),(16, 16)),((1, None),(1, None),(1, None),(16, 16),(32, 32)),
                                      "FRACTAL_NZ","ND","float16",5.5,-4.0,True,"Round","ascend_quant_case_5", "success"))
if __name__ == '__main__':
    ut_case.run("Ascend910A")
