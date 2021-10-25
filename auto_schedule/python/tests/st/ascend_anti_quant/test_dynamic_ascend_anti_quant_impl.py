# # -*- coding:utf-8 -*-
from sch_test_frame.ut import OpUT
from sch_test_frame.common import precision_info
import numpy as np
import tbe


ut_case = OpUT("AscendAntiQuant", "impl.dynamic.ascend_anti_quant", "ascend_anti_quant")

def gen_ascend_anti_quant_case(shape_x, shape_y, ori_shape_x, ori_shape_y, range_x, range_y, format, ori_format, dtype_val, scale, offset, sqrt_mode, kernel_name_val, expect):
    return {"params": [{"shape": shape_x, "dtype": dtype_val, "ori_shape": ori_shape_x, "ori_format": ori_format, "format": format, "range": range_x},
                       {"shape": shape_y, "dtype": "float16", "ori_shape": ori_shape_y, "ori_format": ori_format, "format": format, "range": range_y},
                       scale, offset, sqrt_mode],
            "case_name": kernel_name_val,
            "expect": expect,
            "format_expect": [],
            "support_expect": True}

ut_case.add_case("Ascend910A",
                 gen_ascend_anti_quant_case((-1,-1,-1,-1,32),(-1,-1,-1,-1,16),(-1,-1,-1,-1),(-1,-1,-1,-1),
                                      ((1, None),(1, None),(1, None),(1, None),(32, 32)),((1, None),(1, None),(1, None),(1, None),(16, 16)),
                                      "NC1HWC0","NHWC","int8",1.0,-3.0,False,"ascend_anti_quant_case_1", "success"))
ut_case.add_case("Ascend910A",
                 gen_ascend_anti_quant_case((-1,-1,-1,-1,32),(-1,-1,-1,-1,16),(-1,-1,-1,-1),(-1,-1,-1,-1),
                                      ((1, None),(1, None),(1, None),(1, None),(32, 32)),((1, None),(1, None),(1, None),(1, None),(16, 16)),
                                      "NC1HWC0","NHWC","int8",2.0,0.0,True,"ascend_anti_quant_case_2", "success"))
ut_case.add_case("Ascend910A",
                 gen_ascend_anti_quant_case((-1,-1,-1,-1,32),(-1,-1,-1,-1,16),(-1,-1,-1,-1),(-1,-1,-1,-1),
                                      ((1, None),(1, None),(1, None),(1, None),(32, 32)),((1, None),(1, None),(1, None),(1, None),(16, 16)),
                                      "NC1HWC0","NHWC","int8",3.0,-4.0,True,"ascend_anti_quant_case_3", "success"))

ut_case.add_case("Ascend910A",
                 gen_ascend_anti_quant_case((-1,-1,16,32),(-1,-1,16,16),(-1,-1),(-1,-1),
                                      ((1, None),(1, None),(16, 16),(32, 32)),((1, None),(1, None),(16, 16),(16, 16)),
                                      "NC1HWC0","ND","int8",5.5,-4.3,True,"ascend_anti_quant_case_4", "success"))
ut_case.add_case("Ascend910A",
                 gen_ascend_anti_quant_case((-1,-1,-1,16,32),(-1,-1,-1,16,16),(-1,-1,-1),(-1,-1,-1),
                                      ((1, None),(1, None),(1, None),(16, 16),(32, 32)),((1, None),(1, None),(1, None),(16, 16),(16, 16)),
                                      "NC1HWC0","ND","int8",5.5,-4.0,True,"ascend_anti_quant_case_5", "success"))
                                      
if __name__ == '__main__':
    with tbe.common.context.op_context.OpContext("dynamic"):
        ut_case.run("Ascend910A")