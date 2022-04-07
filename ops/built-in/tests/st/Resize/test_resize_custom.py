from impl.dynamic.resize import resize
import tbe
from te import platform as cce_conf
from impl.dynamic.resize import check_supported
from impl.dynamic.resize import op_select_format


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
        with tbe.common.context.op_context.OpContext("dynamic"):
            resize(input_x, input_roi, None, input_sizes, output_y,
                   coordinate_transformation_mode=coordinate_transformation_mode,
                   mode=mode)
            resize(input_x, input_roi, input_scales, None, output_y,
                   coordinate_transformation_mode=coordinate_transformation_mode,
                   mode=mode)

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

    for coordinate_transformation_mode in coordinate_transformation_mode_all:
        with tbe.common.context.op_context.OpContext("dynamic"):
            resize(input_x, input_roi, None, input_sizes, output_y,
                   coordinate_transformation_mode=coordinate_transformation_mode,
                   mode="nearest")
            resize(input_x, input_roi, input_scales, None, output_y,
                   coordinate_transformation_mode=coordinate_transformation_mode,
                   mode="nearest")


def test_check_supported_op_select_format():
    soc_version_all = ("Ascend910A", "SD3403")
    input_x_all = ({"shape": [1, 1, 5, 5], "ori_shape": [1, 1, 5, 5]},
                   {"shape": [1, 1, 5, 5, 5], "ori_shape": [1, 1, 5, 5, 5]})
    output_y_all = ({"shape": [1, 1, 6, 6], "ori_shape": [1, 1, 6, 6]},
                    {"shape": [1, 1, 6, 6, 6], "ori_shape": [1, 1, 6, 6, 6]})
    mode_name_all = ("nearest", "linear")
    roi, scales, sizes = None, None, None
    for soc, input_x, output_y, mode_name in zip(soc_version_all, input_x_all, output_y_all, mode_name_all):
        cce_conf.te_set_version(soc)
        check_supported(input_x, roi, scales, sizes, output_y, mode=mode_name)
        op_select_format(input_x, roi, scales, sizes, output_y, mode=mode_name)


if __name__ == '__main__':
    soc_version = cce_conf.get_soc_spec("SOC_VERSION")
    cce_conf.te_set_version("Ascend910")
    reload_check_support()
    test_check_supported_op_select_format()
    cce_conf.te_set_version(soc_version)
