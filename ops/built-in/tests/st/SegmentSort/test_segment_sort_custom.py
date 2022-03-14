import te
from te.platform.cce_conf import te_set_version
from impl.segment_sort import segment_sort, op_select_format


class SegmentSortParams:

    def __init__(self, data_num, k_num,
                 data_type="float16", index_type="float16",
                 ai_core_num=32, core_align_num=1984, core_min_num=7936,
                 merge_channel=4, tail_num=16, pro_data_num=8):
        self.data_num = data_num
        self.k_num = k_num
        self.data_type = data_type
        self.index_type = index_type
        self.ai_core_num = ai_core_num
        self.core_align_num = core_align_num
        self.core_min_num = core_min_num
        self.merge_channel = merge_channel
        self.tail_num = tail_num
        self.pro_data_num = pro_data_num


def add_case(params):
    data_format = "ND"
    input_shape_0 = (params.data_num,)
    input_shape_1 = (2048,)
    result_data_num = (params.data_num + params.ai_core_num - 1) // params.ai_core_num
    result_data_num = (result_data_num + params.core_align_num - 1) // params.core_align_num * params.core_align_num
    if result_data_num < params.core_min_num:
        result_data_num = params.core_min_num
    ai_core_num = (params.data_num + result_data_num - 1) // result_data_num
    if ai_core_num > params.merge_channel:
        ai_core_num = (ai_core_num + params.merge_channel - 1) // params.merge_channel * params.merge_channel
    result_data_num = result_data_num + params.tail_num
    output_shape_0 = (ai_core_num, result_data_num, params.pro_data_num)
    params = \
        [
            {"shape": input_shape_0, "format": data_format, "dtype": params.data_type,
             "ori_shape": input_shape_0, "ori_format": data_format},
            {"shape": input_shape_1, "format": data_format, "dtype": params.index_type,
             "ori_shape": input_shape_1, "ori_format": data_format},
            {"shape": output_shape_0, "format": data_format, "dtype": params.data_type,
             "ori_shape": output_shape_0, "ori_format": data_format},
            params.k_num,
        ]
    return params


def reload_check_support():
    """
    reload_check_support to improve cov
    """
    soc_version = te.platform.cce_conf.get_soc_spec("SOC_VERSION")
    te_set_version("Ascend920A", "VectorCore")
    params_920_0 = SegmentSortParams(12288 * 5, 12288 * 5, "float16", "int32", 48, 32, 12288, 4, 32, 4)
    case_2 = add_case(params_920_0)
    op_select_format(*case_2)
    segment_sort(*case_2)
    te_set_version(soc_version)


if __name__ == '__main__':
    reload_check_support()