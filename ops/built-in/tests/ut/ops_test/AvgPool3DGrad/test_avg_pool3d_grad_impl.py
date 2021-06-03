#!/usr/bin/env python
# -*- coding: UTF-8 -*-
# Description : UT test for Conv3DBackpropInput Dynamic
from op_test_frame.ut import OpUT


ut_case = OpUT("AvgPool3DGrad", "impl.dynamic.avg_pool3d_grad", "avg_pool3d_grad")
case_list = []


# Define Utility function
def _gen_data_case(case, expect, case_name_val, support_expect=True):
    return {"params": case,
            "case_name": case_name_val,
            "expect": expect,
            "format_expect": [],
            "support_expect": support_expect}


# def avg_pool3d_grad(orig_input_shape,
#                     grads,
#                     filter,
#                     output,
#                     ksize,
#                     strides,
#                     pads,
#                     ceil_mode=False,
#                     count_include_pad=True,
#                     divisor_override=0,
#                     data_format="NDHWC",
#                     kernel_name="avg_pool3d_grad"):

# test_avgpool3dgrad_succ_dynamic
case1 = [{'ori_shape': (5,), 'shape': (5,), 'format': "ND",'ori_format': 'ND', 'dtype': 'int32', 'range': ((5,5),)},
         {'ori_shape': (16,2,3,-1,1), 'shape': (16,2,1,3,-1,16), 'format': "NDC1HWC0", 'ori_format': 'NDHWC', 'dtype': 'float16',
            'range': ((16,16),(2,2),(1,1),(3,3),(3,40),(16,16))},
         {'ori_shape': (1,5,1,16,1), 'shape': (5,1,16,16), 'format': "FRACTAL_Z_3D", 'ori_format': 'DHWCN', 'dtype': 'float16',
            'range':((5,5),(1,1),(16,16),(16,16))},
         {'ori_shape': (16,12,12,-1,1), 'shape': (16,12,1,12,-1,16), 'format': "NDC1HWC0", 'ori_format': 'NDHWC', 'dtype': 'float16',
            'range': ((16,16), (12,12), (1,1), (12,12), (12,294), (16,16))},
         (1,1,5,1,1),
         (1,10,4,2,1),
         (-1,-1,-1,-1,-1,-1),
         False,
         False,
         0,
         "NDHWC"]
case2 = [{'ori_shape': (5,), 'shape': (5,), 'format': "ND",'ori_format': 'ND', 'dtype': 'int32', 'range': ((5,5),)},
         {'ori_shape': (16,2,-1,6,1), 'shape': (16,2,1,-1,6,16), 'format': "NDC1HWC0", 'ori_format': 'NDHWC', 'dtype': 'float16',
            'range': ((16,16),(2,2),(1,1),(1,10),(6,6),(16,16))},
         {'ori_shape': (1,5,1,16,1), 'shape': (5,1,16,16), 'format': "FRACTAL_Z_3D", 'ori_format': 'DHWCN', 'dtype': 'float16',
            'range':((5,5),(1,1),(16,16),(16,16))},
         {'ori_shape': (16,12,-1,12,1), 'shape': (16,12,1,-1,12,16), 'format': "NDC1HWC0", 'ori_format': 'NDHWC', 'dtype': 'float16',
            'range': ((16,16), (12,12), (1,1), (1,32), (12,12), (16,16))},
         (1,1,5,1,1),
         (1,10,4,2,1),
         (-1,-1,-1,-1,-1,-1),
         False,
         False,
         0,
         "NDHWC"]
case3 = [{'ori_shape': (5,), 'shape': (5,), 'format': "ND",'ori_format': 'ND', 'dtype': 'int32', 'range': ((5,5),)},
         {'ori_shape': (16,-1,3,6,1), 'shape': (16,-1,1,3,6,16), 'format': "NDC1HWC0", 'ori_format': 'NDHWC', 'dtype': 'float16',
            'range': ((16,16),(2,20),(1,1),(3,3),(6,6),(16,16))},
         {'ori_shape': (1,5,1,16,1), 'shape': (5,1,16,16), 'format': "FRACTAL_Z_3D", 'ori_format': 'DHWCN', 'dtype': 'float16',
            'range':((5,5),(1,1),(16,16),(16,16))},
         {'ori_shape': (16,-1,12,12,1), 'shape': (16,-1,1,12,12,16), 'format': "NDC1HWC0", 'ori_format': 'NDHWC', 'dtype': 'float16',
            'range': ((16,16), (10,120), (1,1), (12,12), (12,12), (16,16))},
         (1,1,5,1,1),
         (1,10,4,2,1),
         (-1,-1,-1,-1,-1,-1),
         False,
         False,
         0,
         "NDHWC"]
case4 = [{'ori_shape': (5,), 'shape': (5,), 'format': "ND",'ori_format': 'ND', 'dtype': 'int32', 'range': ((5,5),)},
         {'ori_shape': (-1,2,3,6,1), 'shape': (-1,2,1,3,6,16), 'format': "NDC1HWC0", 'ori_format': 'NDHWC', 'dtype': 'float16',
            'range': ((1,120),(2,2),(1,1),(3,3),(6,6),(16,16))},
         {'ori_shape': (1,5,1,16,1), 'shape': (5,1,16,16), 'format': "FRACTAL_Z_3D", 'ori_format': 'DHWCN', 'dtype': 'float16',
            'range':((5,5),(1,1),(16,16),(16,16))},
         {'ori_shape': (-1,12,12,12,1), 'shape': (-1,12,1,12,12,16), 'format': "NDC1HWC0", 'ori_format': 'NDHWC', 'dtype': 'float16',
            'range': ((1,120), (12,12), (1,1), (12,12), (12,12), (16,16))},
         (1,1,5,1,1),
         (1,10,4,2,1),
         (-1,-1,-1,-1,-1,-1),
         False,
         False,
         0,
         "NDHWC"]
case5 = [{'ori_shape': (5,), 'shape': (5,), 'format': "ND",'ori_format': 'ND', 'dtype': 'int32', 'range': ((5,5),)},
         {'ori_shape': (1,1,4,4,4), 'shape': (1,4,1,4,4,16), 'format': "NDC1HWC0", 'ori_format': 'NCDHW', 'dtype': 'float16',
            'range': ((1,1),(4,4),(1,1),(4,4),(4,4),(16,16))},
         {'ori_shape': (2,2,2,1,1), 'shape': (8,1,16,16), 'format': "FRACTAL_Z_3D", 'ori_format': 'DHWCN', 'dtype': 'float16',
            'range':((8,8),(1,1),(16,16),(16,16))},
         {'ori_shape': (1,1,5,5,5), 'shape': (1,5,1,5,5,16), 'format': "NDC1HWC0", 'ori_format': 'NCDHW', 'dtype': 'float16',
            'range': ((1,1), (5,5), (1,1), (5,5), (5,5), (16,16))},
         (1,1,2,2,2),
         (1,1,1,1,1),
         (0,0,0,0,0,0),
         False,
         False,
         0,
         "NCDHW"]
case6 = [{'ori_shape': (5,), 'shape': (5,), 'format': "ND",'ori_format': 'ND', 'dtype': 'int32', 'range': ((5,5),)},
         {'ori_shape': (1,1,4,4,4), 'shape': (1,1,4,4,4), 'format': "NDC1HWC0", 'ori_format': 'NCDHW', 'dtype': 'float16',
            'range': ((1,120),(1,1),(4,4),(4,4),(4,4))},
         {'ori_shape': (2,2,2,1,1), 'shape': (8,1,16,16), 'format': "FRACTAL_Z_3D", 'ori_format': 'DHWCN', 'dtype': 'float16',
            'range':((8,8),(1,1),(16,16),(16,16))},
         {'ori_shape': (1,1,5,5,5), 'shape': (1,5,1,5,5,16), 'format': "NDC1HWC0", 'ori_format': 'NCDHW', 'dtype': 'float16',
            'range': ((1,1), (5,5), (1,1), (5,5), (5,5), (16,16))},
         (1,2,2,2,1),
         (1,1,1,1,1),
         (0,0,0,0,0,0),
         False,
         False,
         0,
         "NCDHW"]
case7 = [{'ori_shape': (5,), 'shape': (5,), 'format': "ND",'ori_format': 'ND', 'dtype': 'int32', 'range': ((5,5),)},
         {'ori_shape': (1,1,4,4,4), 'shape': (1,4,1,4,4,16), 'format': "NDC1HWC0", 'ori_format': 'DHWCN', 'dtype': 'float16',
            'range': ((1,1),(4,4),(1,1),(4,4),(4,4),(16,16))},
         {'ori_shape': (2,2,2,1,1), 'shape': (8,1,16,16), 'format': "FRACTAL_Z_3D", 'ori_format': 'DHWCN', 'dtype': 'float16',
            'range':((8,8),(1,1),(16,16),(16,16))},
         {'ori_shape': (1,1,5,5,5), 'shape': (1,5,1,5,5,16), 'format': "NDC1HWC0", 'ori_format': 'NCDHW', 'dtype': 'float16',
            'range': ((1,1), (5,5), (1,1), (5,5), (5,5), (16,16))},
         (1,2,2,2,1),
         (1,1,1,1,1),
         (0,0,0,0,0,0),
         False,
         False,
         0,
         "NCDHW"]
case8 = [{'ori_shape': (5,), 'shape': (5,), 'format': "ND",'ori_format': 'ND', 'dtype': 'int32', 'range': ((5,5),)},
         {'ori_shape': (1,1,4,4,4), 'shape': (1,4,1,4,4,16), 'format': "NDC1HWC0", 'ori_format': 'NCDHW', 'dtype': 'float16',
            'range': ((1,1),(4,4),(1,1),(4,4),(4,4),(16,16))},
         {'ori_shape': (2,2,2,1,1), 'shape': (8,1,16,16), 'format': "FRACTAL_Z_3D", 'ori_format': 'DHWCN', 'dtype': 'float16',
            'range':((8,8),(1,1),(16,16),(16,16))},
         {'ori_shape': (1,1,5,5,5), 'shape': (1,5,1,5,5,16), 'format': "NDC1HWC0", 'ori_format': 'NCDHW', 'dtype': 'float16',
            'range': ((1,1), (5,5), (1,1), (5,5), (5,5), (16,16))},
         (1,2,2,2,1),
         (1,1,1,1,1),
         (0,0,0,0,0,0),
         True,
         False,
         0,
         "NCDHW"]
case9 = [{'ori_shape': (5,), 'shape': (5,), 'format': "ND",'ori_format': 'ND', 'dtype': 'int32', 'range': ((5,5),)},
         {'ori_shape': (1,1,4,4,4), 'shape': (1,4,1,4,4,16), 'format': "NDC1HWC0", 'ori_format': 'NCDHW', 'dtype': 'float16',
            'range': ((1,1),(4,4),(1,1),(4,4),(4,4),(16,16))},
         {'ori_shape': (2,2,2,1,1), 'shape': (8,1,16,16), 'format': "FRACTAL_Z_3D", 'ori_format': 'DHWCN', 'dtype': 'float16',
            'range':((8,8),(1,1),(16,16),(16,16))},
         {'ori_shape': (1,1,5,5,5), 'shape': (1,5,1,5,5,16), 'format': "NDC1HWC0", 'ori_format': 'NCDHW', 'dtype': 'float16',
            'range': ((1,1), (5,5), (1,1), (5,5), (5,5), (16,16))},
         (1,2,2,2,1),
         (1,1,1,1,1),
         (0,0,0,0,0,0),
         False,
         True,
         0,
         "NCDHW"]
case10 = [{'ori_shape': (5,), 'shape': (5,), 'format': "ND",'ori_format': 'ND', 'dtype': 'int32', 'range': ((5,5),)},
         {'ori_shape': (1,1,4,4,4), 'shape': (1,4,1,4,4,16), 'format': "NDC1HWC0", 'ori_format': 'NCDHW', 'dtype': 'float16',
            'range': ((1,1),(4,4),(1,1),(4,4),(4,4),(16,16))},
         {'ori_shape': (2,2,2,1,1), 'shape': (8,1,16,16), 'format': "FRACTAL_Z_3D", 'ori_format': 'DHWCN', 'dtype': 'float16',
            'range':((8,8),(1,1),(16,16),(16,16))},
         {'ori_shape': (1,1,5,5,5), 'shape': (1,5,1,5,5,16), 'format': "NDC1HWC0", 'ori_format': 'NCDHW', 'dtype': 'float16',
            'range': ((1,1), (5,5), (1,1), (5,5), (5,5), (16,16))},
         (1,2,2,2,1),
         (1,1,1,1,1),
         (0,0,0,0,0,0),
         False,
         False,
         1,
         "NCDHW"]
case11 = [{'ori_shape': (5,), 'shape': (5,), 'format': "ND",'ori_format': 'ND', 'dtype': 'int32', 'range': ((5,5),)},
         {'ori_shape': ((-2),), 'shape': (1,4,1,4,4,16), 'format': "NDC1HWC0", 'ori_format': 'NCDHW', 'dtype': 'float16',
            'range': ((1,1),(4,4),(1,1),(4,4),(4,4),(16,16))},
         {'ori_shape': (2,2,2,1,1), 'shape': (8,1,16,16), 'format': "FRACTAL_Z_3D", 'ori_format': 'DHWCN', 'dtype': 'float16',
            'range':((8,8),(1,1),(16,16),(16,16))},
         {'ori_shape': (1,1,5,5,5), 'shape': (1,5,1,5,5,16), 'format': "NDC1HWC0", 'ori_format': 'NCDHW', 'dtype': 'float16',
            'range': ((1,1), (5,5), (1,1), (5,5), (5,5), (16,16))},
         (1,2,2,2,1),
         (1,1,1,1,1),
         (0,0,0,0,0,0),
         False,
         False,
         0,
         "NCDHW"]
case12 = [{'ori_shape': (5,), 'shape': (5,), 'format': "ND",'ori_format': 'ND', 'dtype': 'int32', 'range': ((5,5),)},
         {'ori_shape': (1,1,4,4,4), 'shape': (1,4,1,4,4,16), 'format': "NDC1HWC0", 'ori_format': 'NCDHW', 'dtype': 'float16',
            'range': ((1,1),(4,4),(1,1),(4,4),(4,4),(16,16))},
         {'ori_shape': (2,2,2,1,1), 'shape': (8,1,16,16), 'format': "FRACTAL_Z_3D", 'ori_format': 'DHWCN', 'dtype': 'float16',
            'range':((8,8),(1,1),(16,16),(16,16))},
         {'ori_shape': ((-2),), 'shape': (1,5,1,5,5,16), 'format': "NDC1HWC0", 'ori_format': 'NCDHW', 'dtype': 'float16',
            'range': ((1,1), (5,5), (1,1), (5,5), (5,5), (16,16))},
         (1,2,2,2,1),
         (1,1,1,1,1),
         (0,0,0,0,0,0),
         False,
         False,
         0,
         "NCDHW"]
case13 = [{'ori_shape': (5,), 'shape': (5,), 'format': "ND",'ori_format': 'ND', 'dtype': 'int32', 'range': ((5,5),)},
         {'ori_shape': (1,1,4,4), 'shape': (1,4,1,4,4,16), 'format': "NDC1HWC0", 'ori_format': 'NCDHW', 'dtype': 'float16',
            'range': ((1,1),(4,4),(1,1),(4,4),(4,4),(16,16))},
         {'ori_shape': (2,2,2,1,1), 'shape': (8,1,16,16), 'format': "FRACTAL_Z_3D", 'ori_format': 'DHWCN', 'dtype': 'float16',
            'range':((8,8),(1,1),(16,16),(16,16))},
         {'ori_shape': (1,1,5,5,5), 'shape': (1,5,1,5,5,16), 'format': "NDC1HWC0", 'ori_format': 'NCDHW', 'dtype': 'float16',
            'range': ((1,1), (5,5), (1,1), (5,5), (5,5), (16,16))},
         (1,2,2,2,1),
         (1,1,1,1,1),
         (0,0,0,0,0,0),
         False,
         False,
         0,
         "NCDHW"]
case14 = [{'ori_shape': (5,), 'shape': (5,), 'format': "ND",'ori_format': 'ND', 'dtype': 'int32', 'range': ((5,5),)},
         {'ori_shape': (1,1,4,4,4), 'shape': (1,4,1,4,4,16), 'format': "NDC1HWC0", 'ori_format': 'NCDHW', 'dtype': 'float16',
            'range': ((1,1),(4,4),(1,1),(4,4))},
         {'ori_shape': (2,2,2,1,1), 'shape': (8,1,16,16), 'format': "FRACTAL_Z_3D", 'ori_format': 'DHWCN', 'dtype': 'float16',
            'range':((8,8),(1,1),(16,16),(16,16))},
         {'ori_shape': (1,1,5,5,5), 'shape': (1,5,1,5,5,16), 'format': "NDC1HWC0", 'ori_format': 'NCDHW', 'dtype': 'float16',
            'range': ((1,1), (5,5), (1,1), (5,5), (5,5), (16,16))},
         (1,2,2,2,1),
         (1,1,1,1,1),
         (0,0,0,0,0,0),
         False,
         False,
         0,
         "NCDHW"]
case15 = [{'ori_shape': (5,), 'shape': (5,), 'format': "ND",'ori_format': 'ND', 'dtype': 'int32', 'range': ((5,5),)},
         {'ori_shape': (1,1,4,4,4), 'shape': (1,4,1,4,4,16), 'format': "NDC1HWC0", 'ori_format': 'NCDHW', 'dtype': 'float16',
            'range': ((1,1),(4,4),(1,1),(4,4),(4,4),(16,16))},
         {'ori_shape': (2,2,2,1,1), 'shape': (8,1,16,16), 'format': "FRACTAL_Z_3D", 'ori_format': 'DHWCN', 'dtype': 'float16',
            'range':((8,8),(1,1),(16,16),(16,16))},
         {'ori_shape': (1,1,5,5,5), 'shape': (1,5,1,5,5,16), 'format': "NDC1HWC0", 'ori_format': 'NCDHW', 'dtype': 'float16',
            'range': ((1,1), (5,5), (1,1), (5,5), (5,5), (16,16))},
         (1,2,2,2,1),
         (1,1,1,-1,1),
         (0,0,0,0,0,0),
         False,
         False,
         0,
         "NCDHW"]
# Add test Cases
# Params is the input params of the operator.
ut_case.add_case(["Ascend910A"],
                 _gen_data_case(case1, "success", "dynamic_case_w", True))
ut_case.add_case(["Ascend910A"],
                 _gen_data_case(case2, "success", "dynamic_case_h", True))
ut_case.add_case(["Ascend910A"],
                 _gen_data_case(case3, "success", "dynamic_case_d", True))
ut_case.add_case(["Ascend910A"],
                 _gen_data_case(case4, "success", "dynamic_case_n", True))
ut_case.add_case(["Ascend910A"],
                 _gen_data_case(case5, "success", "dynamic_case_all_const", True))
ut_case.add_case(["Ascend910A"],
                 _gen_data_case(case6, RuntimeError, "dynamic_case_range_5d", False))
ut_case.add_case(["Ascend910A"],
                 _gen_data_case(case7, RuntimeError, "dynamic_case_invalid_format", False))
ut_case.add_case(["Ascend910A"],
                 _gen_data_case(case8, RuntimeError, "dynamic_case_unsupport_ceil_mode", False))
ut_case.add_case(["Ascend910A"],
                 _gen_data_case(case9, RuntimeError, "dynamic_case_unsupport_count_include_pad", False))
ut_case.add_case(["Ascend910A"],
                 _gen_data_case(case10, RuntimeError, "dynamic_case_unsupport_divisor_override", False))
ut_case.add_case(["Ascend910A"],
                 _gen_data_case(case11, RuntimeError, "dynamic_case_unsupport_2_grads", False))
ut_case.add_case(["Ascend910A"],
                 _gen_data_case(case12, RuntimeError, "dynamic_case_unsupport_2_dx", False))
ut_case.add_case(["Ascend910A"],
                 _gen_data_case(case13, RuntimeError, "dynamic_case_invalid_grads_shape_dim", False))
ut_case.add_case(["Ascend910A"],
                 _gen_data_case(case14, RuntimeError, "dynamic_case_invalid_grads_range_dim", False))
ut_case.add_case(["Ascend910A"],
                 _gen_data_case(case15, RuntimeError, "dynamic_case_invalid_strides", False))


if __name__ == '__main__':
    import tbe
    with tbe.common.context.op_context.OpContext("dynamic"):
        ut_case.run("Ascend910A")
    exit(0)

