from op_test_frame.ut import OpUT
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


add_case()


if __name__ == '__main__':
    ut_case.run(["Ascend910A", "Ascend310"])
