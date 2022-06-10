from op_test_frame.ut import OpUT
from impl.dynamic.group_norm import check_supported
ut_case = OpUT("GroupNorm", "impl.dynamic.group_norm", "group_norm")


def create_case(input_ori_shape, input_shape, hd_format, ori_format, input_type, num_groups):
    input_x = {"shape": input_shape, "format": hd_format, "dtype": input_type,
               "ori_shape": input_ori_shape, "ori_format": ori_format}
    input_scale = {"shape": [input_ori_shape[1]], "format": "ND", "dtype": input_type,
                   "ori_shape": [input_ori_shape[1]], "ori_format": "ND"}
    input_offset = {"shape": [input_ori_shape[1]], "format": "ND", "dtype": input_type,
                    "ori_shape": [input_ori_shape[1]], "ori_format": "ND"}
    output_y = {"shape": input_shape, "format": hd_format, "dtype": input_type,
                "ori_shape": input_ori_shape, "ori_format": ori_format}
    output_mean = {"shape": [input_ori_shape[1] // num_groups], "format": "ND", "dtype": input_type,
                   "ori_shape": [input_ori_shape[1] // num_groups], "ori_format": "ND"}
    output_var = {"shape": [input_ori_shape[1] // num_groups], "format": "ND", "dtype": input_type,
                  "ori_shape": [input_ori_shape[1] // num_groups], "ori_format": "ND"}
    return [input_x, input_scale, input_offset, output_y, output_mean, output_var, num_groups]


def add_case():
    num_groups = 2
    input_ori_shape = [1, 64, 4, 4]
    input_shape = [1, 4, 4, 4, 16]
    hd_format = "NC1HWC0"
    ori_format = "NCHW"
    input_type = "float32"

    params = create_case(input_ori_shape, input_shape, hd_format, ori_format, input_type, num_groups)
    case1 = {"params": params,
             "case_name": "groupnorm_1",
             "expect": "success",
             "format_expect": [],
             "support_expect": True}
    ut_case.add_case(["Ascend910A", "Ascend310P3"], case1)


def add_case1():
    num_groups = 2
    input_ori_shape = [1, 64, 4, 4]
    input_shape = [1, 64, 4, 4]
    hd_format = "NCHW"
    ori_format = "NCHW"
    input_type = "float32"

    params = create_case(input_ori_shape, input_shape, hd_format, ori_format, input_type, num_groups)
    case2 = {"params": params,
             "case_name": "groupnorm_2",
             "expect": "success",
             "format_expect": [],
             "support_expect": True}
    ut_case.add_case(["Ascend910A", "Ascend310P3"], case2)


def add_check_support(test_args):
    num_groups = 2
    input_ori_shape = [1, 64, 4, 4]
    input_shape = [1, 64, 4, 4]
    hd_format = "NCHW"
    ori_format = "NCHW"
    input_type = "float32"
    data_format = "NCHW"
    epsilon = -0.005

    params = create_case(input_ori_shape, input_shape, hd_format, ori_format, input_type, num_groups)
    params1 = params + [data_format, epsilon]
    params2 = create_case([1, 21, 4, 4], [1, 21, 4, 4], hd_format, ori_format, input_type, num_groups)
    check_supported(*params)
    check_supported(*params1)
    check_supported(*params2)


ut_case.add_cust_test_func("Ascend910A", add_check_support)
add_case()
add_case1()


if __name__ == '__main__':
    ut_case.run(["Ascend910A", "Ascend310P3"])
