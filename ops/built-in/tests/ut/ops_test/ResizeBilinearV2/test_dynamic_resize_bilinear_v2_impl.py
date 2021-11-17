# Copyright 2021 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
ut for resize
"""
import json

from op_test_frame.ut import OpUT


ut_case = OpUT("ResizeBilinearV2", "impl.dynamic.resize_bilinear_v2", "resize_bilinear_v2")

case1 = {"params": [{"shape": (32, 1, 512, 512, 16), "dtype": "float32", "format": "NCHW",
                     "ori_shape": (34, 2, 1, 1, 16),
                     "ori_format": "NCHW", "range": [(1, None), (1, None), (1, None), (1, None), (1, None)]},
                    {"shape": (32, 1, 512, 512, 16), "dtype": "int32", "format": "NCHW",
                     "ori_shape": (34, 2, 1, 1, 16), "ori_format": "NCHW",
                     "range": [(1, None), (1, None), (1, None), (1, None), (1, None)]},
                    {"shape": (32, 1, 512, 512, 16), "dtype": "float32", "format": "NCHW",
                     "ori_shape": (34, 2, 1, 1, 16), "ori_format": "NCHW",
                     "range": [(1, None), (1, None), (1, None), (1, None), (1, None)]}, None, None, None, None,
                    False, False],
         "case_name": "dynamic_resize_bilinear_v2_d_1",
         "expect": "success",
         "format_expect": [],
         "support_expect": True}

case2 = {"params": [{"shape": (-1, -1, -1, -1, 16), "dtype": "float32", "format": "NCHW",
                     "ori_shape": (34, 2, 1, 1, 16), "ori_format": "NCHW",
                     "range": [(1, None), (1, None), (1, None), (1, None), (1, None)]},
                    {"shape": (2,), "dtype": "int32", "format": "NCHW",
                     "ori_shape": (34, 2, 1, 1, 16), "ori_format": "NCHW", "range": [(1, None)]},
                    {"shape": (-1, -1, -1, -1, 16), "dtype": "float32", "format": "NCHW",
                     "ori_shape": (34, 2, 1, 1, 16), "ori_format": "NCHW",
                     "range": [(1, None), (1, None), (1, None), (1, None), (1, None)]}, None, None, None, None,
                    False, False],
         "case_name": "dynamic_resize_bilinear_v2_d_1",
         "expect": "success",
         "format_expect": [],
         "support_expect": True}

case3 = {"params": [{"shape": (-1, -1, -1, -1, 16), "dtype": "float16", "format": "NCHW",
                     "ori_shape": (34, 2, 1, 1, 16), "ori_format": "NCHW",
                     "range": [(1, None), (1, None), (1, None), (1, None), (1, None)]},
                    {"shape": (2,), "dtype": "int32", "format": "NCHW",
                     "ori_shape": (34, 2, 1, 1, 16), "ori_format": "NCHW", "range": [(1, None)]},
                    {"shape": (-1, -1, -1, -1, 16), "dtype": "float32", "format": "NCHW",
                     "ori_shape": (34, 2, 1, 1, 16), "ori_format": "NCHW",
                     "range": [(1, None), (1, None), (1, None), (1, None), (1, None)]}, None, None, None, None,
                    False, False],
         "case_name": "dynamic_resize_bilinear_v2_d_1",
         "expect": "success",
         "format_expect": [],
         "support_expect": True}

case4 = {"params": [{"shape": (-1, -1, -1, -1, 16), "dtype": "float16", "format": "NCHW",
                     "ori_shape": (34, 2, 1, 1, 16), "ori_format": "NCHW",
                     "range": [(1, None), (1, None), (1, None), (1, None), (1, None)]},
                    {"shape": (2,), "dtype": "int32", "format": "NCHW",
                     "ori_shape": (34, 2, 1, 1, 16), "ori_format": "NCHW", "range": [(1, None)]},
                    {"shape": (-1, -1, -1, -1, 16), "dtype": "float16", "format": "NCHW",
                     "ori_shape": (34, 2, 1, 1, 16), "ori_format": "NCHW",
                     "range": [(1, None), (1, None), (1, None), (1, None), (1, None)]}, [16, 16], [8, 8], 0, 0,
                    False, False],
         "case_name": "dynamic_resize_bilinear_v2_d_fp16_to_fp16",
         "expect": "success",
         "format_expect": [],
         "support_expect": True}

case5 = {"params": [{"shape": (-1, -1, -1, -1, 16), "dtype": "float16", "format": "NCHW",
                     "ori_shape": (34, 2, 1, 1, 16), "ori_format": "NCHW",
                     "range": [(1, None), (1, None), (1, None), (1, None), (1, None)]},
                    {"shape": (2,), "dtype": "int32", "format": "NCHW",
                     "ori_shape": (34, 2, 1, 1, 16), "ori_format": "NCHW", "range": [(1, None)]},
                    {"shape": (-1, -1, -1, -1, 16), "dtype": "float16", "format": "NCHW",
                     "ori_shape": (34, 2, 1, 1, 16), "ori_format": "NCHW",
                     "range": [(1, None), (1, None), (1, None), (1, None), (1, None)]}, [16, 16], [8, 8], 0, 0,
                    False, True],
         "case_name": "dynamic_resize_bilinear_v2_d_fp16_to_fp16_5",
         "expect": "success",
         "format_expect": [],
         "support_expect": True}

case6 = {"params": [{"shape": (-1, -1, -1, -1, 16), "dtype": "float16", "format": "NCHW",
                     "ori_shape": (34, 2, 1, 1, 16), "ori_format": "NCHW",
                     "range": [(1, None), (1, None), (1, None), (1, None), (1, None)]},
                    {"shape": (2,), "dtype": "int32", "format": "NCHW",
                     "ori_shape": (34, 2, 1, 1, 16), "ori_format": "NCHW", "range": [(1, None)]},
                    {"shape": (-1, -1, -1, -1, 16), "dtype": "float16", "format": "NCHW",
                     "ori_shape": (34, 2, 1, 1, 16), "ori_format": "NCHW",
                     "range": [(1, None), (1, None), (1, None), (1, None), (1, None)]}, [16, 16], [8, 8], 0, 0,
                    True, False],
         "case_name": "dynamic_resize_bilinear_v2_d_fp16_to_fp16_6",
         "expect": "success",
         "format_expect": [],
         "support_expect": True}


ut_case.add_case("all", case1)
ut_case.add_case("all", case2)
ut_case.add_case("all", case3)
ut_case.add_case("all", case4)
ut_case.add_case("all", case5)
ut_case.add_case("all", case6)


def cmp_data(src_data, dst_data):
    """
    cmp

    Parameters:
    ----------
    src_data: original data
    dst_data: compare data

    Returns
    -------
    compare result
    """
    if isinstance(src_data, dict):
        for key in dst_data:
            if key not in src_data:
                return False
        for key in src_data:
            if key in dst_data:
                cmp_data(src_data[key], dst_data[key])
            else:
                return False
    elif isinstance(src_data, list):
        if len(src_data) != len(dst_data):
            return False
        for src_list, dst_list in zip(sorted(src_data), sorted(dst_data)):
            cmp_data(src_list, dst_list)
    else:
        if str(src_data) != str(dst_data):
            return False

    return True


# pylint: disable=unused-argument
def tune_space_resize_bilinear_v2_case1(test_arg):
    """
    tune_space_resize_bilinear_v2_case1

    Parameters:
    ----------
    test_arg: may be used for te_set_version()

    Returns
    -------
    None
    """
    from impl.dynamic.resize_bilinear_v2 import tune_space_resize_bilinear_v2
    from impl.util.platform_adapter import tik
    aicore_num = tik.Dprofile().get_aicore_num()
    images = {"shape": (34, 2, 1, 1, 16), "dtype": "float16", "format": "NCHW",
              "ori_shape": (34, 2, 1, 1, 16), "ori_format": "NCHW",
              "range": [(1, None), (1, None), (1, None), (1, None), (1, None)]}
    size = {"shape": (2,), "dtype": "int32", "format": "NCHW",
            "ori_shape": (2,), "ori_format": "NCHW", "range": [(1, None)]}
    y = {"shape": (34, 2, 1, 1, 16), "dtype": "float16", "format": "NCHW",
         "ori_shape": (34, 2, 1, 1, 16), "ori_format": "NCHW",
         "range": [(1, None), (1, None), (1, None), (1, None), (1, None)]}
    tune_param_expect = {}
    tune_param_expect["version"] = "1.0.0"
    tune_param_expect["tune_timeout"] = 600
    tune_param_expect["tune_param"] = {}
    tune_param_expect["tune_param"]["param"] = "tiling_key"
    tune_param_expect["tune_param"]["data_type"] = "int64"
    tune_param_expect["tune_param"]["type"] = "list"
    tune_param_expect["tune_param"]["sub_param"] = {}
    tune_param_expect["tune_param"]["sub_param"][0] = {}
    tune_param_expect["tune_param"]["sub_param"][0]["value"] = 999999
    tune_param_expect["tune_param"]["sub_param"][1] = {}
    tune_param_expect["tune_param"]["sub_param"][1]["value"] = 100110
    tune_param_expect["tune_param"]["sub_param"][1]["param_list"] = {}
    tune_param_expect["tune_param"]["sub_param"][1]["param_list"][0] = {}
    tune_param_expect["tune_param"]["sub_param"][1]["param_list"][0]["param"] = "cut_batch_c1_num"
    tune_param_expect["tune_param"]["sub_param"][1]["param_list"][0]["data_type"] = "int64"
    tune_param_expect["tune_param"]["sub_param"][1]["param_list"][0]["type"] = "range"
    tune_param_expect["tune_param"]["sub_param"][1]["param_list"][0]["value"] = [1, aicore_num]
    tune_param_expect["tune_param"]["sub_param"][1]["param_list"][1] = {}
    tune_param_expect["tune_param"]["sub_param"][1]["param_list"][1]["param"] = "cut_height_num"
    tune_param_expect["tune_param"]["sub_param"][1]["param_list"][1]["data_type"] = "int64"
    tune_param_expect["tune_param"]["sub_param"][1]["param_list"][1]["type"] = "range"
    tune_param_expect["tune_param"]["sub_param"][1]["param_list"][1]["value"] = [1, aicore_num]
    tune_param_expect["tune_param"]["sub_param"][1]["param_list"][2] = {}
    tune_param_expect["tune_param"]["sub_param"][1]["param_list"][2]["param"] = "cut_width_num"
    tune_param_expect["tune_param"]["sub_param"][1]["param_list"][2]["data_type"] = "int64"
    tune_param_expect["tune_param"]["sub_param"][1]["param_list"][2]["type"] = "range"
    tune_param_expect["tune_param"]["sub_param"][1]["param_list"][2]["value"] = [1, aicore_num]
    tune_param_expect["tune_param"]["sub_param"][2] = {}
    tune_param_expect["tune_param"]["sub_param"][2]["value"] = 100000
    tune_param_expect["tune_param"]["sub_param"][2]["param_list"] = {}
    tune_param_expect["tune_param"]["sub_param"][2]["param_list"][0] = {}
    tune_param_expect["tune_param"]["sub_param"][2]["param_list"][0]["param"] = "cut_batch_c1_num"
    tune_param_expect["tune_param"]["sub_param"][2]["param_list"][0]["data_type"] = "int64"
    tune_param_expect["tune_param"]["sub_param"][2]["param_list"][0]["type"] = "range"
    tune_param_expect["tune_param"]["sub_param"][2]["param_list"][0]["value"] = [1, aicore_num]
    tune_param_expect["tune_param"]["sub_param"][2]["param_list"][1] = {}
    tune_param_expect["tune_param"]["sub_param"][2]["param_list"][1]["param"] = "cut_height_num"
    tune_param_expect["tune_param"]["sub_param"][2]["param_list"][1]["data_type"] = "int64"
    tune_param_expect["tune_param"]["sub_param"][2]["param_list"][1]["type"] = "range"
    tune_param_expect["tune_param"]["sub_param"][2]["param_list"][1]["value"] = [1, aicore_num]
    tune_param_expect["tune_param"]["sub_param"][2]["param_list"][2] = {}
    tune_param_expect["tune_param"]["sub_param"][2]["param_list"][2]["param"] = "cut_width_num"
    tune_param_expect["tune_param"]["sub_param"][2]["param_list"][2]["data_type"] = "int64"
    tune_param_expect["tune_param"]["sub_param"][2]["param_list"][2]["type"] = "range"
    tune_param_expect["tune_param"]["sub_param"][2]["param_list"][2]["value"] = [1, aicore_num]

    tune_param_actual = tune_space_resize_bilinear_v2(images, size, y, None, None, None, None, False, False,
                                                      "resize_bilinear_v2")
    if not cmp_data(json.loads(tune_param_actual), tune_param_expect):
        raise Exception("Failed to call tune_space_resize_bilinear_v2 in resize_bilinear_v2.")


ut_case.add_cust_test_func(test_func=tune_space_resize_bilinear_v2_case1)


# pylint: disable=unused-argument
def tune_param_check_supported_resize_bilinear_v2_case1(test_arg):
    """
    tune_param_check_supported_resize_bilinear_v2_case1

    Parameters:
    ----------
    test_arg: may be used for te_set_version()

    Returns
    -------
    None
    """
    from impl.dynamic.resize_bilinear_v2 import tune_param_check_supported_resize_bilinear_v2
    from impl.util.platform_adapter import tik
    aicore_num = tik.Dprofile().get_aicore_num()
    images = {"shape": (-1, -1, -1, -1, 16), "dtype": "float16", "format": "NCHW",
              "ori_shape": (34, 2, 1, 1, 16), "ori_format": "NCHW",
              "range": [(1, None), (1, None), (1, None), (1, None), (1, None)]}
    size = {"shape": (2,), "dtype": "int32", "format": "NCHW",
            "ori_shape": (34, 2, 1, 1, 16), "ori_format": "NCHW", "range": [(1, None)]}
    y = {"shape": (-1, -1, -1, -1, 16), "dtype": "float16", "format": "NCHW",
         "ori_shape": (34, 2, 1, 1, 16), "ori_format": "NCHW",
         "range": [(1, None), (1, None), (1, None), (1, None), (1, None)]}
    tune_param = {"version": "1.0.0", "tune_param": {"tiling_key": 999999,
                                                     "cut_batch_c1_num": aicore_num,
                                                     "cut_height_num": aicore_num,
                                                     "cut_width_num": aicore_num}}
    print('tune_param: ', tune_param)
    if not tune_param_check_supported_resize_bilinear_v2(images, size, y, None, None, None, None, False, False,
                                                         "resize_bilinear_v2",
                                                         json.dumps(tune_param)):
        raise Exception("Failed to call tune_param_check_supported_resize_bilinear_v2 in resize_bilinear_v2.")


# pylint: disable=unused-argument
def tune_param_check_supported_resize_bilinear_v2_case2(test_arg):
    """
    tune_param_check_supported_resize_bilinear_v2_case2

    Parameters:
    ----------
    test_arg: may be used for te_set_version()

    Returns
    -------
    None
    """
    from impl.dynamic.resize_bilinear_v2 import tune_param_check_supported_resize_bilinear_v2
    from impl.util.platform_adapter import tik
    aicore_num = tik.Dprofile().get_aicore_num()
    images = {"shape": (-1, -1, -1, -1, 16), "dtype": "float16", "format": "NCHW",
              "ori_shape": (34, 2, 1, 1, 16), "ori_format": "NCHW",
              "range": [(1, None), (1, None), (1, None), (1, None), (1, None)]}
    size = {"shape": (2,), "dtype": "int32", "format": "NCHW",
            "ori_shape": (34, 2, 1, 1, 16), "ori_format": "NCHW", "range": [(1, None)]}
    y = {"shape": (-1, -1, -1, -1, 16), "dtype": "float16", "format": "NCHW",
         "ori_shape": (34, 2, 1, 1, 16), "ori_format": "NCHW",
         "range": [(1, None), (1, None), (1, None), (1, None), (1, None)]}
    tune_param = {"version": "1.0.0", "tune_param": {"tiling_key": 100110,
                                                     "cut_batch_c1_num": aicore_num // 2,
                                                     "cut_height_num": 1,
                                                     "cut_width_num": 1}}
    if not tune_param_check_supported_resize_bilinear_v2(images, size, y, None, None, None, None, False, False,
                                                         "resize_bilinear_v2",
                                                         json.dumps(tune_param)):
        raise Exception("Failed to call tune_param_check_supported_resize_bilinear_v2 in resize_bilinear_v2.")


# pylint: disable=unused-argument
def tune_param_check_supported_resize_bilinear_v2_case3(test_arg):
    """
    tune_param_check_supported_resize_bilinear_v2_case3

    Parameters:
    ----------
    test_arg: may be used for te_set_version()

    Returns
    -------
    None
    """
    from impl.dynamic.resize_bilinear_v2 import tune_param_check_supported_resize_bilinear_v2
    images = {"shape": (-1, -1, -1, -1, 16), "dtype": "float16", "format": "NCHW",
              "ori_shape": (34, 2, 1, 1, 16), "ori_format": "NCHW",
              "range": [(1, None), (1, None), (1, None), (1, None), (1, None)]}
    size = {"shape": (2,), "dtype": "int32", "format": "NCHW",
            "ori_shape": (34, 2, 1, 1, 16), "ori_format": "NCHW", "range": [(1, None)]}
    y = {"shape": (-1, -1, -1, -1, 16), "dtype": "float16", "format": "NCHW",
         "ori_shape": (34, 2, 1, 1, 16), "ori_format": "NCHW",
         "range": [(1, None), (1, None), (1, None), (1, None), (1, None)]}
    tune_param = {"version": "1.0.0", "tune_param": {"tiling_key": 888888,
                                                     "cut_batch_c1_num": 1,
                                                     "cut_height_num": 1,
                                                     "cut_width_num": 1}}
    if tune_param_check_supported_resize_bilinear_v2(images, size, y, None, None, None, None, False, False,
                                                     "resize_bilinear_v2",
                                                     json.dumps(tune_param)):
        raise Exception("Failed to call tune_param_check_supported_resize_bilinear_v2 in resize_bilinear_v2.")


def tune_param_check_supported_resize_bilinear_v2_case4(test_arg):
    """
    tune_param_check_supported_resize_bilinear_v2_case4

    Parameters:
    ----------
    test_arg: may be used for te_set_version()

    Returns
    -------
    None
    """
    from impl.dynamic.resize_bilinear_v2 import tune_param_check_supported_resize_bilinear_v2
    from impl.util.platform_adapter import tik
    aicore_num = tik.Dprofile().get_aicore_num()
    images = {"shape": (-1, -1, -1, -1, 16), "dtype": "float16", "format": "NCHW",
              "ori_shape": (34, 2, 1, 1, 16), "ori_format": "NCHW",
              "range": [(1, None), (1, None), (1, None), (1, None), (1, None)]}
    size = {"shape": (2,), "dtype": "int32", "format": "NCHW",
            "ori_shape": (34, 2, 1, 1, 16), "ori_format": "NCHW", "range": [(1, None)]}
    y = {"shape": (-1, -1, -1, -1, 16), "dtype": "float16", "format": "NCHW",
         "ori_shape": (34, 2, 1, 1, 16), "ori_format": "NCHW",
         "range": [(1, None), (1, None), (1, None), (1, None), (1, None)]}
    tune_param = {"version": "1.0.0", "tune_param": {"tiling_key": 100000,
                                                     "cut_batch_c1_num": aicore_num,
                                                     "cut_height_num": aicore_num,
                                                     "cut_width_num": aicore_num}}
    if tune_param_check_supported_resize_bilinear_v2(images, size, y, None, None, None, None, False, False,
                                                     "resize_bilinear_v2",
                                                     json.dumps(tune_param)):
        raise Exception("Failed to call tune_param_check_supported_resize_bilinear_v2 in resize_bilinear_v2.")


def sync_resize(test_arg):
    import tbe
    from impl.dynamic.sync_resize_bilinear_v2 import SyncResizeBilinearV2
    params = [{"shape": (-1, -1, -1, -1, 16), "dtype": "float16", "format": "NCHW", "ori_shape": (34, 2, 1, 1, 16),
               "ori_format": "NCHW", "range": [(1, None), (1, None), (1, None), (1, None), (1, None)]},
              {"shape": (2,), "dtype": "int32", "format": "NCHW", "ori_shape": (34, 2, 1, 1, 16),
               "ori_format": "NCHW", "range": [(1, None)]}, {"shape": (-1, -1, -1, -1, 16), "dtype": "float16",
                                                             "format": "NCHW", "ori_shape": (34, 2, 1, 1, 16),
                                                             "ori_format": "NCHW",
                                                             "range": [(1, None), (1, None), (1, None),
                                                                       (1, None), (1, None)]},
              [16, 16], [8, 8], 0, 0, True, False, "resize_bilinear_v2"]
    with tbe.common.context.op_context.OpContext("dynamic"):
        obj = SyncResizeBilinearV2(*params)
        obj.resize_bilinear_v2_operator()


ut_case.add_cust_test_func(test_func=tune_param_check_supported_resize_bilinear_v2_case1)
ut_case.add_cust_test_func(test_func=tune_param_check_supported_resize_bilinear_v2_case2)
ut_case.add_cust_test_func(test_func=tune_param_check_supported_resize_bilinear_v2_case3)
ut_case.add_cust_test_func(test_func=tune_param_check_supported_resize_bilinear_v2_case4)
ut_case.add_cust_test_func(test_func=sync_resize)


if __name__ == '__main__':
    ut_case.run("Ascend910A")
