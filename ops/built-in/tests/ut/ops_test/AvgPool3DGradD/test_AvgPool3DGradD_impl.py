#!/usr/bin/env python
# -*- coding: UTF-8 -*-
# Description : UT test for AvgPool3DGradD
from op_test_frame.ut import OpUT

ut_case = OpUT("AvgPool3DGradD",
               "impl.avg_pool3d_grad_d",
               "avg_pool3d_grad_d")

# Define Utility function
def _gen_data_case(case, expect, case_name_val, support_expect=True):
    return {"params": case,
            "case_name": case_name_val,
            "expect": expect,
            "format_expect": [],
            "support_expect": support_expect}

# test_avg_pool3d_grad_d_succ in global mode
case1 = [{'ori_shape': (1, 1, 1, 1, 1), 'shape': (1, 1, 1, 1, 1, 16), 'ori_format': 'NDHWC', 'format': 'NDC1HWC0', 'dtype': 'float16'},
         None,
         None,
         {'ori_shape': (1, 3, 3, 3, 1), 'shape': (1, 3, 1, 3, 3, 16), 'format': "NDC1HWC0", 'ori_format': 'NDHWC', 'dtype': 'float16'},
         (1, 3, 3, 3, 1),
         (1, 3, 3, 3, 1),
         (1, 1, 1, 1, 1),
         (0, 0, 0, 0, 0, 0),
         False,
         False,
         0,
         "NDHWC"]

case2 = [{'ori_shape': (9, 6, 4, 14, 48), 'shape': (9, 6, 3, 4, 14, 16), 'ori_format': 'NDHWC', 'format': 'NDC1HWC0', 'dtype': 'float16'},
         {'ori_shape': (1, 2, 2, 1, 48), 'shape': (12, 1, 16, 16), 'format': "FRACTAL_Z_3D", 'ori_format': 'DHWCN', 'dtype': 'float16'},
         {'ori_shape': (9, 6, 4, 14, 48), 'shape': (9, 6, 3, 4, 14, 16), 'ori_format': 'NDHWC', 'format': 'NDC1HWC0', 'dtype': 'float16'},
         {'ori_shape': (9, 6, 28, 28, 48), 'shape': (9, 6, 3, 28, 28, 16), 'format': "NDC1HWC0", 'ori_format': 'NDHWC', 'dtype': 'float16'},
         (9, 6, 28, 28, 48),
         (1, 1, 2, 2, 1),
         (1, 1, 9, 2, 1),
         (0, 0, 0, 1, 0, 0),
         False,
         False,
         0,
         "NDHWC"]

case3 = [{'ori_shape': (1, 3, 3, 3, 1), 'shape': (1, 3, 1, 3, 3, 16), 'ori_format': 'NDHWC', 'format': 'NDC1HWC0', 'dtype': 'float16'},
         {'ori_shape': (2, 2, 2, 1, 1), 'shape': (8, 1, 16, 16), 'format': "FRACTAL_Z_3D", 'ori_format': 'DHWCN', 'dtype': 'float16'},
         {'ori_shape': (1, 3, 3, 3, 1), 'shape': (1, 3, 1, 3, 3, 16), 'ori_format': 'NDHWC', 'format': 'NDC1HWC0', 'dtype': 'float16'},
         {'ori_shape': (1, 5, 5, 5, 1), 'shape': (1, 5 ,1, 5, 5, 16), 'format': "NDC1HWC0", 'ori_format': 'NDHWC', 'dtype': 'float16'},
         (1,5,5,5,1),
         (1,2,2,2,1),
         (1,2,2,2,1),
         (0,0,0,0,0,0),
         True,
         False,
         0,
         "NDHWC"]

case4 = [{'ori_shape': (1, 3, 3, 3, 1), 'shape': (1, 3, 1, 3, 3, 16), 'ori_format': 'NDHWC', 'format': 'NDC1HWC0', 'dtype': 'float16'},
         {'ori_shape': (2, 2, 2, 1, 1), 'shape': (8, 1, 16, 16), 'format': "FRACTAL_Z_3D", 'ori_format': 'DHWCN', 'dtype': 'float16'},
         None,
         {'ori_shape': (1, 5, 5, 5, 1), 'shape': (1, 5 ,1, 5, 5, 16), 'format': "NDC1HWC0", 'ori_format': 'NDHWC', 'dtype': 'float16'},
         (1,5,5,5,1),
         (1,2,2,2,1),
         (1,2,2,2,1),
         (1,1,1,1,1,1),
         False,
         True,
         0,
         "NDHWC"]

case5 = [{'ori_shape': (1, 3, 3, 3, 1), 'shape': (1, 3, 1, 3, 3, 16), 'ori_format': 'NDHWC', 'format': 'NDC1HWC0', 'dtype': 'float16'},
         {'ori_shape': (2, 2, 2, 1, 1), 'shape': (8, 1, 16, 16), 'format': "FRACTAL_Z_3D", 'ori_format': 'DHWCN', 'dtype': 'float16'},
         {'ori_shape': (1, 3, 3, 3, 1), 'shape': (1, 3, 1, 3, 3, 16), 'ori_format': 'NDHWC', 'format': 'NDC1HWC0', 'dtype': 'float16'},
         {'ori_shape': (1, 5, 5, 5, 1), 'shape': (1, 5 ,1, 5, 5, 16), 'format': "NDC1HWC0", 'ori_format': 'NDHWC', 'dtype': 'float16'},
         (1,5,5,5,1),
         (2,2,2),
         (1,2,2,2,1),
         (1,1,1,1,1,1),
         False,
         False,
         0,
         "NDHWC"]

case6 = [{'ori_shape': (1, 3, 3, 3, 1), 'shape': (1, 3, 1, 3, 3, 16), 'ori_format': 'NDHWC', 'format': 'NDC1HWC0', 'dtype': 'float16'},
         {'ori_shape': (2, 2, 2, 1, 1), 'shape': (8, 1, 16, 16), 'format': "FRACTAL_Z_3D", 'ori_format': 'DHWCN', 'dtype': 'float16'},
         {'ori_shape': (1, 3, 3, 3, 1), 'shape': (1, 3, 1, 3, 3, 16), 'ori_format': 'NDHWC', 'format': 'NDC1HWC0', 'dtype': 'float16'},
         {'ori_shape': (1, 5, 5, 5, 1), 'shape': (1, 5 ,1, 5, 5, 16), 'format': "NDC1HWC0", 'ori_format': 'NDHWC', 'dtype': 'float16'},
         (1,5,5,5,1),
         (1,2,2,2,1),
         (2,2,2),
         (1,1,1,1,1,1),
         False,
         False,
         0,
         "NDHWC"]

case7 = [{'ori_shape': (1, 1, 1, 1, 1), 'shape': (1, 1, 1, 1, 1, 16), 'ori_format': 'NDHWC', 'format': 'NDC1HWC0', 'dtype': 'float16'},
         None,
         None,
         {'ori_shape': (1, 3, 3, 3, 1), 'shape': (1, 3, 1, 3, 3, 16), 'format': "NDC1HWC0", 'ori_format': 'NDHWC', 'dtype': 'float16'},
         (1, 3, 3, 3, 1),
         (1, 3, 3, 3, 1),
         (1, 1, 1, 1, 1),
         (0, 0, 0, 0, 0, 0),
         False,
         False,
         2,
         "NDHWC"]

case8 = [{'ori_shape': (1, 1, 1, 1, 1), 'shape': (1, 1, 1, 1, 1, 16), 'ori_format': 'NDHWC', 'format': 'NDC1HWC0', 'dtype': 'float16'},
         None,
         None,
         {'ori_shape': (1, 3, 3, 3, 1), 'shape': (1, 3, 1, 3, 3, 16), 'format': "NDC1HWC0", 'ori_format': 'NDHWC', 'dtype': 'float16'},
         (1, 3, 3, 3, 1),
         (1, 3, 3, 3, 1),
         (1, 1, 1, 1, 1),
         (0, 0, 0, 0, 0, 0),
         False,
         True,
         0,
         "NDHWC"]

case9 = [{'ori_shape': (1, 2, 1, 1, 1), 'shape': (1, 1, 1, 1, 1, 16), 'ori_format': 'NDHWC', 'format': 'NDC1HWC0', 'dtype': 'float16'},
         None,
         None,
         {'ori_shape': (1, 3, 3, 3, 1), 'shape': (1, 3, 1, 3, 3, 16), 'format': "NDC1HWC0", 'ori_format': 'NDHWC', 'dtype': 'float16'},
         (1, 3, 3, 3, 1),
         (1, 3, 3, 3, 1),
         (1, 1, 1, 1, 1),
         (0, 0, 0, 0, 0, 0),
         False,
         False,
         0,
         "NDHWC"]

case10 = [{'ori_shape': (1, 1, 3, 3, 3), 'shape': (1, 3, 1, 3, 3, 16), 'ori_format': 'NDHWC', 'format': 'NDC1HWC0', 'dtype': 'float16'},
          {'ori_shape': (2, 2, 2, 1, 1), 'shape': (8, 1, 16, 16), 'format': "FRACTAL_Z_3D", 'ori_format': 'DHWCN', 'dtype': 'float16'},
          {'ori_shape': (1, 1, 3, 3, 3), 'shape': (1, 3, 1, 3, 3, 16), 'ori_format': 'NDHWC', 'format': 'NDC1HWC0', 'dtype': 'float16'},
          {'ori_shape': (1, 1, 5, 5, 5), 'shape': (1, 5 ,1, 5, 5, 16), 'format': "NDC1HWC0", 'ori_format': 'NDHWC', 'dtype': 'float16'},
          (1,5,5,5,1),
          (2,1,2,2,2),
          (1,1,2,2,2),
          (1,1,1,1,1,1),
          False,
          False,
          0,
          "NCDHW"]

case11 = [{'ori_shape': (1, 1, 3, 3, 3), 'shape': (1, 3, 1, 3, 3, 16), 'ori_format': 'NDHWC', 'format': 'NDC1HWC0', 'dtype': 'float16'},
          {'ori_shape': (2, 2, 2, 1, 1), 'shape': (8, 1, 16, 16), 'format': "FRACTAL_Z_3D", 'ori_format': 'DHWCN', 'dtype': 'float16'},
          {'ori_shape': (1, 1, 3, 3, 3), 'shape': (1, 3, 1, 3, 3, 16), 'ori_format': 'NDHWC', 'format': 'NDC1HWC0', 'dtype': 'float16'},
          {'ori_shape': (1, 1, 5, 5, 5), 'shape': (1, 5 ,1, 5, 5, 16), 'format': "NDC1HWC0", 'ori_format': 'NDHWC', 'dtype': 'float16'},
          (1,5,5,5,1),
          (1,1,2,2,2),
          (2,1,2,2,2),
          (1,1,1,1,1,1),
          False,
          False,
          0,
          "NCDHW"]

case12 = [{'ori_shape': (1, 1, 3, 3, 3), 'shape': (1, 3, 1, 3, 3, 16), 'ori_format': 'NDHWC', 'format': 'NDC1HWC0', 'dtype': 'float16'},
          {'ori_shape': (2, 2, 2, 1, 1), 'shape': (8, 1, 16, 16), 'format': "FRACTAL_Z_3D", 'ori_format': 'DHWCN', 'dtype': 'float16'},
          {'ori_shape': (1, 1, 3, 3, 3), 'shape': (1, 3, 1, 3, 3, 16), 'ori_format': 'NDHWC', 'format': 'NDC1HWC0', 'dtype': 'float16'},
          {'ori_shape': (1, 1, 5, 5, 5), 'shape': (1, 5 ,1, 5, 5, 16), 'format': "NDC1HWC0", 'ori_format': 'NDHWC', 'dtype': 'float16'},
          (1,5,5,5,1),
          (1,1,2,2,2),
          (1,1,2,2,2),
          (1,1,1,1,1),
          False,
          False,
          0,
          "NCDHW"]


# Add test Cases
# Params is the input params of the operator.
ut_case.add_case(["Ascend910A"],
                 _gen_data_case(case1, "success", "static_global_base_case", True))
ut_case.add_case(["Ascend910A"],
                 _gen_data_case(case2, "success", "static_cube_base_case", True))
ut_case.add_case(["Ascend910A"],
                 _gen_data_case(case3, "success", "static_cube_ceil_mode_case", True))
ut_case.add_case(["Ascend910A"],
                 _gen_data_case(case4, "success", "static_cube_count_include_pads_case", True))
ut_case.add_case(["Ascend910A"],
                 _gen_data_case(case5, RuntimeError, "static_cube_invalid_ksize_dim_case", False))
ut_case.add_case(["Ascend910A"],
                 _gen_data_case(case6, RuntimeError, "static_cube_invalid_strides_case", False))
ut_case.add_case(["Ascend910A"],
                 _gen_data_case(case7, "success", "static_global_divisor_override_case", True))
ut_case.add_case(["Ascend910A"],
                 _gen_data_case(case8, "success", "static_global_count_include_pad_case", True))
ut_case.add_case(["Ascend910A"],
                 _gen_data_case(case9, RuntimeError, "static_global_invaild_shape_case", True))
ut_case.add_case(["Ascend910A"],
                 _gen_data_case(case10, RuntimeError, "static_cube_invalid_ksize_val_case", False))
ut_case.add_case(["Ascend910A"],
                 _gen_data_case(case11, RuntimeError, "static_cube_invalid_strides_val_case", False))
ut_case.add_case(["Ascend910A"],
                 _gen_data_case(case12, RuntimeError, "static_cube_invaild_pads_dims_case", False))

if __name__ == '__main__':
    ut_case.run()
    exit(0)
