# # -*- coding:utf-8 -*-
from op_test_frame.ut import OpUT
from impl.multi_merge import op_select_format, multi_merge
import te
from te.platform.cce_conf import te_set_version
ut_case = OpUT("MultiMerge", None, None)


def add_case(input_shape, k_num, pro_repeat_num, data_type, num, merge_channel=4):
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
    case_temp = {
        "params": [
            {"shape": input_shape, "format": data_format, "dtype": data_type,
             "ori_shape": input_shape, "ori_format": data_format},
            {"shape": output_shape_0, "format": data_format, "dtype": data_type,
             "ori_shape": output_shape_0, "ori_format": data_format},
            {"shape": output_shape_1, "format": data_format, "dtype": index_dtype,
             "ori_shape": output_shape_1, "ori_format": data_format},
            k_num,
            include_index,
        ],
        "case_name": "multi_merge_{}".format(num),
        "expect": "success",
        "format_expect": [],
        "support_expect": True}
    return case_temp


case_0 = add_case((32, 1808412, 8), 1808412, 16, "float16", 0)
case_1 = add_case((2, 1808432, 8), 1808412, 16, "float16", 1)
ut_case.add_case(["Ascend910A"], case_0)
ut_case.add_case(["Ascend910A"], case_1)


def test_op_select_format(_):
    params = case_0.get("params")
    op_select_format(*params)
    soc_version = te.platform.cce_conf.get_soc_spec("SOC_VERSION")
    te_set_version("Ascend920A", "VectorCore")
    case_2 = add_case((32, 3277584, 4), 1808412, 16, "float16", 2)
    case_3 = add_case((2, 1808432, 4), 1808412, 16, "float16", 3)
    case_4 = add_case((1, 1808432, 4), 1808412, 16, "float16", 4)

    params_2 = case_2.get("params")
    op_select_format(*params_2)
    multi_merge(*params_2)

    params_3 = case_3.get("params")
    multi_merge(*params_3)

    params_4 = case_4.get("params")
    multi_merge(*params_4)

    te_set_version(soc_version)


ut_case.add_cust_test_func("Ascend910A", test_op_select_format)


if __name__ == '__main__':
    ut_case.run(["Ascend910A", "Ascend920A"])
    exit(0)
