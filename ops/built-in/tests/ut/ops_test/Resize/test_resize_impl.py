from op_test_frame.ut import OpUT
from te import platform as cce_conf
from impl.dynamic.resize import check_supported
from impl.dynamic.resize import op_select_format
ut_case = OpUT("Resize", "impl.dynamic.resize", "resize")


def add_case():
    coordinate_transformation_mode_0 = "pytorch_half_pixel"
    coordinate_transformation_mode_1 = "align_corners"
    coordinate_transformation_mode_2 = "asymmetric"
    cubic_coeff_a = -0.75
    exclude_outside = 0
    extrapolation_value = 0.0
    mode_name_0 = "nearest"
    mode_name_1 = "linear"
    input_ori_shape = [1, 1, 4, 4]
    input_shape = [1, 1, 4, 4, 16]
    hd_format = "NC1HWC0"
    ori_format = "NCHW"
    input_type = "float32"
    scales_shape = [4]
    scales_type = "float32"
    sizes_shape = [4]
    sizes_type = "int32"
    output_ori_shape = [1, 1, 8, 8]
    output_shape = [1, 1, 8, 8, 16]
    output_type = "float32"

    input_x = {"shape": input_shape, "format": hd_format, "dtype": input_type,
               "ori_shape": input_ori_shape, "ori_format": ori_format}
    input_roi = None
    input_scales = {"shape": scales_shape, "format": ori_format, "dtype": scales_type,
                    "ori_shape": scales_shape, "ori_format": ori_format}
    input_sizes = {"shape": sizes_shape, "format": ori_format, "dtype": sizes_type,
                   "ori_shape": sizes_shape, "ori_format": ori_format}
    output_y = {"shape": output_shape, "format": hd_format, "dtype": output_type,
                "ori_shape": output_ori_shape, "ori_format": ori_format}
    case1 = {"params": [input_x,
                        input_roi,
                        input_scales,
                        None,
                        output_y,
                        coordinate_transformation_mode_0,
                        cubic_coeff_a,
                        exclude_outside,
                        extrapolation_value,
                        mode_name_0],
             "case_name": "resize_1",
             "expect": "success",
             "format_expect": [],
             "support_expect": True}
    case2 = {"params": [input_x,
                        input_roi,
                        None,
                        input_sizes,
                        output_y,
                        coordinate_transformation_mode_1,
                        cubic_coeff_a,
                        exclude_outside,
                        extrapolation_value,
                        mode_name_1],
             "case_name": "resize_1",
             "expect": "success",
             "format_expect": [],
             "support_expect": True}
    case2 = {"params": [input_x,
                        input_roi,
                        None,
                        input_sizes,
                        output_y,
                        coordinate_transformation_mode_2,
                        cubic_coeff_a,
                        exclude_outside,
                        extrapolation_value,
                        mode_name_1],
             "case_name": "resize_1",
             "expect": "success",
             "format_expect": [],
             "support_expect": True}
    ut_case.add_case(["Ascend910A", "Ascend310"], case1)
    ut_case.add_case(["Ascend910A", "Ascend310"], case2)


def add_case_3d():
    coordinate_transformation_mode_0 = "pytorch_half_pixel"
    coordinate_transformation_mode_1 = "align_corners"
    coordinate_transformation_mode_2 = "asymmetric"
    cubic_coeff_a = -0.75
    exclude_outside = 0
    extrapolation_value = 0.0
    mode_name_0 = "nearest"
    input_ori_shape = [1, 1, 4, 4, 4]
    input_shape = [1, 1, 4, 4, 4, 16]
    hd_format = "NDC1HWC0"
    ori_format = "NCDHW"
    input_type = "float32"
    scales_shape = [4]
    scales_type = "float32"
    sizes_shape = [4]
    sizes_type = "int32"
    output_ori_shape = [1, 1, 8, 8, 8]
    output_shape = [1, 1, 8, 8, 8, 16]
    output_type = "float32"

    input_x = {"shape": input_shape, "format": hd_format, "dtype": input_type,
               "ori_shape": input_ori_shape, "ori_format": ori_format}
    input_roi = None
    input_scales = {"shape": scales_shape, "format": ori_format, "dtype": scales_type,
                    "ori_shape": scales_shape, "ori_format": ori_format}
    input_sizes = {"shape": sizes_shape, "format": ori_format, "dtype": sizes_type,
                   "ori_shape": sizes_shape, "ori_format": ori_format}
    output_y = {"shape": output_shape, "format": hd_format, "dtype": output_type,
                "ori_shape": output_ori_shape, "ori_format": ori_format}
    case1 = {"params": [input_x,
                        input_roi,
                        input_scales,
                        None,
                        output_y,
                        coordinate_transformation_mode_0,
                        cubic_coeff_a,
                        exclude_outside,
                        extrapolation_value,
                        mode_name_0],
             "case_name": "resize_1",
             "expect": "success",
             "format_expect": [],
             "support_expect": True}

    nearest_mode = "round_prefer_ceil"
    case2 = {"params": [input_x,
                        input_roi,
                        None,
                        input_sizes,
                        output_y,
                        coordinate_transformation_mode_2,
                        cubic_coeff_a,
                        exclude_outside,
                        extrapolation_value,
                        mode_name_0,
                        nearest_mode],
             "case_name": "resize_1",
             "expect": "success",
             "format_expect": [],
             "support_expect": True}

    ut_case.add_case(["Ascend910A"], case1)
    ut_case.add_case(["Ascend910A"], case2)


add_case()
add_case_3d()


def test_op_select_format(_):
    soc_version_all = ("Ascend910A", "SD3403")
    input_x_all = ({"shape": [1, 1, 5, 5], "ori_shape": [1, 1, 5, 5]},
                   {"shape": [1, 1, 5, 5, 5], "ori_shape": [1, 1, 5, 5, 5]})
    output_y_all = ({"shape": [1, 1, 6, 6], "ori_shape": [1, 1, 6, 6]},
                    {"shape": [1, 1, 6, 6, 6], "ori_shape": [1, 1, 6, 6, 6]})
    mode_name_all = ("nearest", "linear")
    roi, scales, sizes = None, None, None
    soc_version = cce_conf.get_soc_spec("SOC_VERSION")
    for soc, input_x, output_y, mode_name in zip(soc_version_all, input_x_all, output_y_all, mode_name_all):
        cce_conf.te_set_version(soc)
        check_supported(input_x, roi, scales, sizes, output_y, mode=mode_name)
        op_select_format(input_x, roi, scales, sizes, output_y, mode=mode_name)
    cce_conf.te_set_version(soc_version)

ut_case.add_cust_test_func("Ascend910A", test_op_select_format)

if __name__ == '__main__':
    ut_case.run(["Ascend910A", "Ascend310"])
