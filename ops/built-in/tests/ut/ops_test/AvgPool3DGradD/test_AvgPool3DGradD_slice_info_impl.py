from op_test_frame.ut import OpUT


ut_case = OpUT("AvgPool3DGradD",
               "impl.avg_pool3d_grad_d",
               "get_op_support_info")

# Define Utility function
def _gen_data_case(case, expect, case_name_val, support_expect=True):
    return {"params": case,
            "case_name": case_name_val,
            "expect": expect,
            "format_expect": [],
            "support_expect": support_expect}

op_info_case1 = [{'ori_shape': (1, 1, 1, 1, 1), 'shape': (1, 1, 1, 1, 1, 16), 'ori_format': 'NDHWC', 'format': 'NDC1HWC0', 'dtype': 'float16'},
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

op_info_case2 = [{'ori_shape': (9, 6, 4, 14, 48), 'shape': (9, 6, 3, 4, 14, 16), 'ori_format': 'NDHWC', 'format': 'NDC1HWC0', 'dtype': 'float16'},
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

op_info_case3 = [{'ori_shape': (9, 6, 4, 14, 48), 'shape': (9, 6, 3, 4, 14, 16), 'ori_format': 'NDHWC', 'format': 'NDC1HWC0', 'dtype': 'float16'},
         {'ori_shape': (1, 2, 2, 1, 48), 'shape': (12, 1, 16, 16), 'format': "FRACTAL_Z_3D", 'ori_format': 'DHWCN', 'dtype': 'float16'},
         None,
         {'ori_shape': (9, 6, 28, 28, 48), 'shape': (9, 6, 3, 28, 28, 16), 'format': "NDC1HWC0", 'ori_format': 'NDHWC', 'dtype': 'float16'},
         (9, 6, 28, 28, 48),
         (1, 1, 2, 2, 1),
         (1, 1, 9, 2, 1),
         (0, 0, 0, 1, 0, 0),
         False,
         False,
         0,
         "NDHWC"]

op_info_case4 = [{'ori_shape': (6, 4, 14, 48, 9), 'shape': (9, 6, 3, 4, 14, 16), 'ori_format': 'DHWCN', 'format': 'NDC1HWC0', 'dtype': 'float16'},
         {'ori_shape': (1, 2, 2, 1, 48), 'shape': (12, 1, 16, 16), 'format': "FRACTAL_Z_3D", 'ori_format': 'DHWCN', 'dtype': 'float16'},
         None,
         {'ori_shape': (9, 6, 28, 28, 48), 'shape': (9, 6, 3, 28, 28, 16), 'format': "NDC1HWC0", 'ori_format': 'NDHWC', 'dtype': 'float16'},
         (9, 6, 28, 28, 48),
         (1, 1, 2, 2, 1),
         (1, 1, 9, 2, 1),
         (0, 0, 0, 1, 0, 0),
         False,
         False,
         0,
         "NDHWC"]

op_info_case5 = [{'ori_shape': (9, 6, 4, 14, 48), 'shape': (9, 6, 3, 4, 14, 16), 'ori_format': 'NDHWC', 'format': 'NDC1HWC0', 'dtype': 'float16'},
         {'ori_shape': (1, 2, 2, 1, 48), 'shape': (12, 1, 16, 16), 'format': "FRACTAL_Z_3D", 'ori_format': 'DHWCN', 'dtype': 'float16'},
         None,
         {'ori_shape': (6, 4, 14, 48, 9), 'shape': (9, 6, 3, 28, 28, 16), 'format': "NDC1HWC0", 'ori_format': 'DHWCN', 'dtype': 'float16'},
         (9, 6, 28, 28, 48),
         (1, 1, 2, 2, 1),
         (1, 1, 9, 2, 1),
         (0, 0, 0, 1, 0, 0),
         False,
         False,
         0,
         "DHWCN"]

ut_case.add_case(["Ascend910A"],
        _gen_data_case(op_info_case1, 'success', "op_support_info_global_case", True))
ut_case.add_case(["Ascend910A"],
        _gen_data_case(op_info_case2, 'success', "op_support_info_base_case", True))
ut_case.add_case(["Ascend910A"],
        _gen_data_case(op_info_case3, 'success', "op_support_info_no_mean_matrix_case", True))
ut_case.add_case(["Ascend910A"],
        _gen_data_case(op_info_case4, RuntimeError, "op_support_info_invalid_grads_case", False))
ut_case.add_case(["Ascend910A"],
        _gen_data_case(op_info_case5, RuntimeError, "op_support_info_invalid_data_format_case", False))


if __name__ == '__main__':
    ut_case.run()
    exit(0)
