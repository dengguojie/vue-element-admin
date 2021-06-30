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
# ============================================================================
"""
op_cfg_file ut test
"""
import unittest
import os


# pylint: disable=invalid-name,bad-classmethod-argument,too-many-locals,too-many-branches,unused-variable
# pylint: disable=too-many-statements,consider-using-sys-exit
cur_file_path = __file__
ini_path = os.path.dirname(cur_file_path)  # llt/ops/llt_new/ut/cfg_format_check
ini_path = os.path.dirname(ini_path)  # llt/ops/llt_new/ut
ini_path = os.path.dirname(ini_path)  # llt/ops/llt_new
ini_path = os.path.dirname(ini_path)  # llt/ops
ini_path = os.path.dirname(ini_path)  # llt
ini_path = os.path.dirname(ini_path)  #
ini_path = os.path.join(ini_path, "ops/built-in/tbe/op_info_cfg/ai_core")  #

class Test_OpConfig(unittest.TestCase):
    """Test_OpConfig"""
    def setUp(self):
        # 每个测试用例执行之前做操作
        pass

    def tearDown(self):
        # 每个测试用例执行之后做操作
        pass

    @classmethod
    def tearDownClass(self):
        # 必须使用 @ classmethod装饰器, 所有test运行完后运行一次
        print("")
        print("---------------------------------------------------")

    @classmethod
    def setUpClass(self):
        # 必须使用@classmethod 装饰器,所有test运行前运行一次
        print("---------------------------------------------------")

    def test_op_config(self):
        """test_op_config"""
        allow_types = set(["int4", "int8", "uint8", "int16", "uint16", "int32", "uint32",
                           "int64", "uint64", "float16", "float32",
                           "float", "bool", "double"])
        allow_input_info_keys = set(["name", "dtype", "format", "shape", "reshapeType",
                                     "needCompile", "shapesType", "valueDepend",
                                     "unknownshape_format", "valueDepend"])
        allow_reshape_types = set(["N", "C", "H", "W", "NC", "CN", "NH", "NW", "CH", "NCH", "NHW", "NWC"])
        allow_output_info_keys = set(["name", "dtype", "format", "shape", "reshapeType",
                                      "needCompile", "shapesType", "unknownshape_format"])
        allow_param_types = set(["required", "optional", "dynamic"])
        allow_op_pattern = set(["reduce", "broadcast", "formatAgnostic"])
        allow_attr_type = set(["bool", "int", "listInt", "float", "str", "listFloat",
                               "listListInt", "type", "listBool"])
        allow_op_key = set(["dynamicFormat.flag", "op.pattern", "compute.cost", "partial.flag", "async.flag",
                            "binfile.name", "kernel.name",
                            "opFile.value", "opInterface.value", "heavyOp.flag", "precision_reduce.flag",
                            "needCheckSupport.flag", "dynamicShapeSupport.flag", "slicePattern.value", "dynamicCompileStatic.flag",
                            "dynamicRankSupport.flag"])
        allow_attr_info_key = set(["type", "value", "paramType", "defaultValue"])
        allow_formats = set(['FRACTAL_NZ', 'FRACTAL_Z', 'ND', 'FRACTAL_ZN_LSTM',
                             'NC1HWC0', 'C1HWNCoC0', 'NCHW', 'NHWC',
                             'NDC1HWC0', 'FRACTAL_Z_3D', 'NC1HWC0_C04', 'HWCN', 'CHWN',
                             'NDHWC', 'DHWCN', 'NCDHW', 'NULL', 'FRACTAL_Z_C04', 'FRACTAL_ZN_RNN', 'ND_RNN_BIAS'])

        def check_op_key_info(file_name, op_type, op_key_infos):
            """
            check_op_key_info
            """
            attr_list = []
            attr_info_map = {}
            check_error_msg = []
            format_dtype_list = []
            format_list = []
            has_format = False
            has_unkown_shape_format = False
            is_dynamic = False
            is_pattern = False
            for op_key_info in op_key_infos:
                info_key = op_key_info[:op_key_info.index("=")].strip()
                info_value = op_key_info[op_key_info.index("=") + 1:].strip()
                # if op_type == "MaxPoolGrad":
                #     print(op_type, op_key_info, info_key, info_value)
                if info_key.startswith("input"):
                    input_info_key = info_key[info_key.index(".") + 1:]
                    if input_info_key == "dtype":
                        dt_list = [x.strip() for x in info_value.split(",")]
                        not_allow_dt = []
                        for dt in dt_list:
                            if dt not in allow_types:
                                not_allow_dt.append(dt)
                        if not_allow_dt:
                            check_error_msg.append(
                                "%s check failed, not allowed dtype: %s" % (op_type, ",".join(not_allow_dt)))
                        format_dtype_list.append(dt_list)
                    elif input_info_key == "format":
                        has_format = True
                        format_dtype_list.append([x.strip() for x in info_value.split(",")])
                        format_list.append([x.strip() for x in info_value.split(",")])
                    elif input_info_key == "unknownshape_format":
                        has_unkown_shape_format = True
                        format_dtype_list.append([x.strip() for x in info_value.split(",")])
                        format_list.append([x.strip() for x in info_value.split(",")])
                    elif input_info_key == "paramType":
                        if info_value not in allow_param_types:
                            check_error_msg.append(
                                "%s check failed, paramType is not allowed: %s" % (op_type, op_key_info))
                    elif input_info_key == "reshapeType":
                        if info_value not in allow_reshape_types:
                            check_error_msg.append(
                                "%s check failed, reshapeType is not allowed: %s" % (op_type, op_key_info))
                    elif input_info_key == "shape":
                        pass
                    else:
                        if input_info_key not in allow_input_info_keys:
                            check_error_msg.append(
                                "%s check failed, input can't cfg this info: %s" % (op_type, op_key_info))
                elif info_key.startswith("output"):
                    output_info_key = info_key[info_key.index(".") + 1:]
                    if output_info_key == "dtype":
                        dt_list = [x.strip() for x in info_value.split(",")]
                        not_allow_dt = []
                        for dt in dt_list:
                            if dt not in allow_types:
                                not_allow_dt.append(dt)
                        if not_allow_dt:
                            check_error_msg.append(
                                "%s check failed, not allowed dtype: %s" % (op_type, ",".join(not_allow_dt)))
                        format_dtype_list.append(dt_list)
                    elif output_info_key == "format":
                        has_format = True
                        format_dtype_list.append([x.strip() for x in info_value.split(",")])
                        format_list.append([x.strip() for x in info_value.split(",")])
                    elif output_info_key == "unknownshape_format":
                        has_unkown_shape_format = True
                        format_dtype_list.append([x.strip() for x in info_value.split(",")])
                        format_list.append([x.strip() for x in info_value.split(",")])
                    elif output_info_key == "paramType":
                        if info_value not in allow_param_types:
                            check_error_msg.append(
                                "%s check failed, not allowed paramType: %s" % (op_type, op_key_info))
                    elif output_info_key == "reshapeType":
                        if info_value not in allow_reshape_types:
                            check_error_msg.append(
                                "%s check failed, not allowed reshapeType: %s" % (op_type, op_key_info))
                    elif output_info_key == "shape":
                        pass
                    else:
                        if output_info_key not in allow_output_info_keys:
                            check_error_msg.append(
                                "%s check failed, output can't cfg this info: %s" % (op_type, op_key_info))
                elif info_key.startswith("attr"):
                    if info_key == "attr.list":
                        attr_list = sorted([x.strip() for x in info_value.split(",")])
                        # if op_type == "MaxPoolGrad":
                        #     print("MaxPoolGrad", attr_list)
                        #     print("MaxPoolGrad", info_value)
                        #     print("MaxPoolGrad", op_key_info)
                    else:
                        attr_name = info_key[5:info_key.index(".")]
                        attr_info = {}
                        if attr_name not in attr_info_map.keys():
                            attr_info_map[attr_name] = attr_info
                        else:
                            attr_info = attr_info_map[attr_name]
                        attr_info_key = info_key[info_key.index(".") + 1:].strip()
                        attr_info[attr_info_key] = info_value
                        if attr_info_key not in allow_attr_info_key:
                            check_error_msg.append(
                                "%s check failed, attr info can't cfg this info: %s" % (op_type, op_key_info))
                        if attr_info_key == "type":
                            if info_value not in allow_attr_type:
                                check_error_msg.append(
                                    "%s check failed, attr type is not allowed: %s" % (op_type, op_key_info))
                        elif attr_info_key == "paramType":
                            if info_value not in allow_param_types:
                                check_error_msg.append(
                                    "%s check failed, attr paramType is not allowed: %s" % (op_type, op_key_info))
                        elif attr_info_key == "value":
                            pass
                        else:
                            pass
                elif info_key.startswith("op.pattern"):
                    is_pattern = True
                    if info_value not in allow_op_pattern:
                        check_error_msg.append(
                            "%s check failed, op.pattern value is not allowed: %s" % (op_type, op_key_info))
                elif info_key.startswith("dynamicFormat.flag"):
                    is_dynamic = True
                else:
                    if info_key not in allow_op_key:
                        check_error_msg.append(
                            "%s check failed, op info can't config this info: %s" % (op_type, op_key_info))

            if attr_list != sorted(attr_info_map.keys()):
                check_error_msg.append("%s check failed, attr.list is not match attr info" % op_type)
            if len(format_list) > 0:
                for formats in format_list:
                    for format_item in formats:
                        if format_item not in allow_formats:
                            check_error_msg.append(
                                "%s check failed: format %s is not allow" % (op_type, format_item))
            if len(format_dtype_list) > 0:
                f_d_table_len = len(format_dtype_list[0])
                for f_d in format_dtype_list:
                    if len(f_d) != f_d_table_len:
                        check_error_msg.append("%s check failed: format and dtype len is not match" % op_type)
            if is_dynamic and has_format:
                check_error_msg.append("%s check failed: configed dynamic.flag don't need config format" % op_type)
            if is_pattern and has_format:
                check_error_msg.append("%s check failed: configed op.pattern don't need config format" % op_type)
            if len(check_error_msg) != 0:
                raise AssertionError(
                    "check op cfg failed! error file: %s msg: %s" % (file_name, "\n".join(check_error_msg)))

        def check_cfg_info(c_f_file_path):
            with open(c_f_file_path) as c_f:
                lines = c_f.readlines()
            op_key_info = None
            op_type = None
            op_types = []
            check_valid = True
            line_num = 0
            for line in lines:
                line_num += 1
                line = line.strip()
                # if line_num
                # print(line_num, line)
                if line.startswith("["):
                    if op_type is not None:
                        if op_type not in op_types:
                            op_types.append(op_type)
                        else:
                            raise AssertionError("check op cfg failed! error file: %s msg: %s is not unique" % (
                                c_f_file_path, op_type))
                        if not check_op_key_info(c_f_file_path, op_type, op_key_info):
                            check_valid = False
                    op_type = line[line.index("[") + 1: line.index("]")]
                    op_key_info = []
                else:
                    op_key_info.append(line)
            return check_valid

        for soc_dir in os.listdir(ini_path):
            soc_path = os.path.join(ini_path, soc_dir)
            for file_name in os.listdir(soc_path):
                if file_name.endswith(".ini"):
                    file_path = os.path.join(soc_path, file_name)
                    with self.subTest("test cfg file %s" % file_name):
                        print(file_name)
                        check_cfg_info(file_path)

if __name__ == "__main__":
    unittest.main()
    exit(0)
