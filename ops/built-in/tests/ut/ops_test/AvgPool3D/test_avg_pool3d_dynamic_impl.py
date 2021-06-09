#!/usr/bin/env python
# -*- coding: UTF-8 -*-
# Description : UT test for Conv3DBackpropInput Dynamic
from op_test_frame.ut import OpUT


ut_case = OpUT("AvgPool3D", "impl.dynamic.avg_pool3d", "avg_pool3d")
case_list = []


# Define Utility function
def _gen_data_case(case, expect, case_name_val, support_expect=True):
    return {"params": case,
            "case_name": case_name_val,
            "expect": expect,
            "format_expect": [],
            "support_expect": support_expect}


# def avg_pool3d(x,
#                filter,
#                y,
#                ksize,
#                strides,
#                pads,
#                ceil_mode=False,
#                count_inclue_pad=True,
#                divisor_override=0,
#                data_format="NDHWC",
#                kernel_name="avg_pool3d")

# test_avgpool3dgrad_succ_dynamic
case1 = [{'ori_shape': (-1,16,5,5,5), 'shape': (-1,5,1,5,5,16), 'format': "NDC1HWC0", 'ori_format': 'NCDHW', 'dtype': 'float16',
            'range': ((1,3),(5,5),(1,1),(5,5),(5,5),(16,16))},
         {'ori_shape': (2,2,2,1,16), 'shape': (8,1,16,16), 'format': "FRACTAL_Z_3D", 'ori_format': 'DHWCN', 'dtype': 'float16',
            'range':((8,8),(1,1),(16,16),(16,16))},
         {'ori_shape': (-1,16,4,4,4), 'shape': (-1,4,1,4,4,16), 'format': "NDC1HWC0", 'ori_format': 'NCDHW', 'dtype': 'float16',
            'range': ((1,3),(4,4),(1,1),(4,4),(4,4),(16,16))},
         (1,1,2,2,2),
         (1,1,1,1,1),
         (0,0,0,0,0,0),
         False,
         False,
         0,
         "NCDHW"]
case2 = [{'ori_shape': (2,16,-1,5,5), 'shape': (2,-1,1,5,5,16), 'format': "NDC1HWC0", 'ori_format': 'NCDHW', 'dtype': 'float16',
            'range': ((2,2),(1,10),(1,1),(5,5),(5,5),(16,16))},
         {'ori_shape': (2,2,2,1,16), 'shape': (8,1,16,16), 'format': "FRACTAL_Z_3D", 'ori_format': 'DHWCN', 'dtype': 'float16',
            'range':((8,8),(1,1),(16,16),(16,16))},
         {'ori_shape': (2,16,-1,4,4), 'shape': (2,-1,1,4,4,16), 'format': "NDC1HWC0", 'ori_format': 'NCDHW', 'dtype': 'float16',
            'range': ((2,2),(1,10),(1,1),(4,4),(4,4),(16,16))},
         (1,1,2,2,2),
         (1,1,1,1,1),
         (0,0,0,0,0,0),
         False,
         False,
         0,
         "NCDHW"]
case3 = [{'ori_shape': (2,16,5,-1,5), 'shape': (2,5,1,-1,5,16), 'format': "NDC1HWC0", 'ori_format': 'NCDHW', 'dtype': 'float16',
            'range': ((2,2),(5,5),(1,1),(1,10),(5,5),(16,16))},
         {'ori_shape': (2,2,2,1,16), 'shape': (8,1,16,16), 'format': "FRACTAL_Z_3D", 'ori_format': 'DHWCN', 'dtype': 'float16',
            'range':((8,8),(1,1),(16,16),(16,16))},
         {'ori_shape': (2,16,4,-1,4), 'shape': (2,4,1,-1,4,16), 'format': "NDC1HWC0", 'ori_format': 'NCDHW', 'dtype': 'float16',
            'range': ((2,2),(4,4),(1,1),(1,10),(4,4),(16,16))},
         (1,1,2,2,2),
         (1,1,1,1,1),
         (0,0,0,0,0,0),
         False,
         False,
         0,
         "NCDHW"]
case4 = [{'ori_shape': (2,16,5,5,-1), 'shape': (2,5,1,5,-1,16), 'format': "NDC1HWC0", 'ori_format': 'NCDHW', 'dtype': 'float16',
            'range': ((2,2),(5,5),(1,1),(5,5),(1,10),(16,16))},
         {'ori_shape': (2,2,2,1,16), 'shape': (8,1,16,16), 'format': "FRACTAL_Z_3D", 'ori_format': 'DHWCN', 'dtype': 'float16',
            'range':((8,8),(1,1),(16,16),(16,16))},
         {'ori_shape': (2,16,4,4,-1), 'shape': (2,4,1,4,-1,16), 'format': "NDC1HWC0", 'ori_format': 'NCDHW', 'dtype': 'float16',
            'range': ((2,2),(4,4),(1,1),(4,4),(1,10),(16,16))},
         (1,1,2,2,2),
         (1,1,1,1,1),
         (0,0,0,0,0,0),
         False,
         False,
         0,
         "NCDHW"]
case5 = [{'ori_shape': (2,16,5,5,-1), 'shape': (2,5,1,5,-1,16), 'format': "NDC1HWC0", 'ori_format': 'NCDHW', 'dtype': 'float16',
            'range': ((2,2),(5,5),(1,1),(5,5),(1,10),(16,16))},
         {'ori_shape': (2,2,2,1,16), 'shape': (8,1,16,16), 'format': "FRACTAL_Z_3D", 'ori_format': 'DHWCN', 'dtype': 'float16',
            'range':((8,8),(1,1),(16,16),(16,16))},
         {'ori_shape': (2,16,5,5,-1), 'shape': (2,5,1,5,-1,16), 'format': "NDC1HWC0", 'ori_format': 'NCDHW', 'dtype': 'float16',
            'range': ((2,2),(5,5),(1,1),(5,5),(1,10),(16,16))},
         (1,1,2,2,2),
         (1,1,1,1,1),
         (-1,-1,-1,-1,-1,-1),
         False,
         False,
         0,
         "NCDHW"]

# Add test Cases
# Params is the input params of the operator.
ut_case.add_case(["Ascend910A"],
                 _gen_data_case(case1, "success", "dynamic_case_n_VALID", True))
ut_case.add_case(["Ascend910A"],
                 _gen_data_case(case2, "success", "dynamic_case_d_VALID", True))
ut_case.add_case(["Ascend910A"],
                 _gen_data_case(case3, "success", "dynamic_case_h_VALID", True))
ut_case.add_case(["Ascend910A"],
                 _gen_data_case(case4, "success", "dynamic_case_w_VALID", True))
ut_case.add_case(["Ascend910A"],
                 _gen_data_case(case5, "success", "dynamic_case_w_SAME", True))

if __name__ == '__main__':
    import tbe
    with tbe.common.context.op_context.OpContext("dynamic"):
        ut_case.run()
    exit(0)

