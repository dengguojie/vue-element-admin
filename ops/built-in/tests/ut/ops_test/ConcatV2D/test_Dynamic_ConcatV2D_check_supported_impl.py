from op_test_frame.ut import OpUT

ut_case = OpUT("ConcatD", "impl.dynamic.concat_v2_d", "concat_v2_d")

def test_op_check_supported(test_arg):
    from impl.dynamic.concat_v2_d import check_supported
    case1 = {"params": [[{"shape": (128, 128, 128, 128), "dtype": "float16", "format": "NHWC", "ori_shape": (128, 128, 128, 128),"ori_format": "NHWC"},
                     {"shape": (128, 128, 128, 128), "dtype": "float16", "format": "NHWC", "ori_shape": (128, 128, 128, 128),"ori_format": "NHWC"}],
                    {"shape": (128, 128, 128, 128), "dtype": "float16", "format": "NHWC", "ori_shape": (128, 128, 128, 128),"ori_format": "NHWC"},
                    1],
         "case_name": "dynamic_concat_v2_d_check_supported_1",
         "expect": "success",
         "format_expect": [],
         "support_expect": True}

    input_datas = case1.get("params")[0]
    output_data = case1.get("params")[1]
    concat_v2_dim = case1.get("params")[2]
    kernel_name = case1.get("case_name")
    res, _ = check_supported(input_datas, output_data, concat_v2_dim, kernel_name)


def test_op_check_supported_dynamic(test_arg):
    from impl.dynamic.concat_v2_d import check_supported
    case1 = {"params": [[{"shape": (128, 128, 128, 128), "dtype": "float16", "format": "NHWC", "ori_shape": (128, 128, 128, 128),"ori_format": "NHWC"},
                     {"shape": (-2,), "dtype": "float16", "format": "NHWC", "ori_shape": (-2,),"ori_format": "NHWC"}],
                    {"shape": (128, 128, 128, 128), "dtype": "float16", "format": "NHWC", "ori_shape": (128, 128, 128, 128),"ori_format": "NHWC"},
                    1],
         "case_name": "dynamic_concat_v2_d_check_supported_2",
         "expect": "success",
         "format_expect": [],
         "support_expect": True}

    input_datas = case1.get("params")[0]
    output_data = case1.get("params")[1]
    concat_v2_dim = case1.get("params")[2]
    kernel_name = case1.get("case_name")
    res, _ = check_supported(input_datas, output_data, concat_v2_dim, kernel_name)


ut_case.add_cust_test_func(test_func=test_op_check_supported)
ut_case.add_cust_test_func(test_func=test_op_check_supported_dynamic)
