import te
from te.platform.cce_conf import te_set_version
from impl.multi_merge import multi_merge, op_select_format


def add_case(input_shape, k_num, pro_repeat_num, data_type, merge_channel=4):
    data_format = "ND"
    index_dtype = "int32"
    if input_shape[0] <= merge_channel:
        include_index = True
        output_shape_0 = (k_num, )
        output_shape_1 = (k_num, )
    else:
        include_index = False
        data_num = input_shape[1]
        result_data_num = data_num * merge_channel
        k_align_num = (k_num + pro_repeat_num - 1) // pro_repeat_num * pro_repeat_num
        if k_align_num < result_data_num:
            result_data_num = k_align_num
        result_data_num = result_data_num + pro_repeat_num
        ai_core_num = input_shape[0] // merge_channel
        if ai_core_num > merge_channel:
            ai_core_num = (ai_core_num + merge_channel - 1) // merge_channel * merge_channel
        output_shape_0 = (ai_core_num, result_data_num, input_shape[2])
        output_shape_1 = (1, )

    params = [
        {"shape": input_shape, "format": data_format, "dtype": data_type,
         "ori_shape": input_shape, "ori_format": data_format},
        {"shape": output_shape_0, "format": data_format, "dtype": data_type,
         "ori_shape": output_shape_0, "ori_format": data_format},
        {"shape": output_shape_1, "format": data_format, "dtype": index_dtype,
         "ori_shape": output_shape_1, "ori_format": data_format},
        k_num,
        include_index,
    ]
    return params


def reload_check_support():
    """
    reload_check_support to improve cov
    """
    soc_version = te.platform.cce_conf.get_soc_spec("SOC_VERSION")
    te_set_version("Ascend920A", "VectorCore")
    case_2 = add_case((32, 3277584, 4), 1808412, 16, "float16")
    case_3 = add_case((2, 1808432, 4), 1808412, 16, "float16")
    case_4 = add_case((1, 1808432, 4), 1808412, 16, "float16")
    op_select_format(*case_2)
    multi_merge(*case_2)
    multi_merge(*case_3)
    multi_merge(*case_4)
    te_set_version(soc_version)


if __name__ == '__main__':
    reload_check_support()
