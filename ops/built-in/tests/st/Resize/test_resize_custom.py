from impl.dynamic.resize import resize


def reload_check_support():
    """
    reload_check_support to improve cov
    """
    mode_name_all = ("nearest", "linear", "linear")
    coordinate_transformation_mode_all = ("pytorch_half_pixel", "align_corners", "asymmetric")
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
    for coordinate_transformation_mode, mode in zip(coordinate_transformation_mode_all, mode_name_all):
        resize(input_x, input_roi, None, input_sizes, output_y,
               coordinate_transformation_mode=coordinate_transformation_mode,
               mode=mode)
        resize(input_x, input_roi, input_scales, None, output_y,
               coordinate_transformation_mode=coordinate_transformation_mode,
               mode=mode)


if __name__ == '__main__':
    reload_check_support()
