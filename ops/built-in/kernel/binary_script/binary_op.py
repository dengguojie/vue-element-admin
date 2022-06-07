# Copyright (c) Huawei Technologies Co., Ltd. 2022. All rights reserved.
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
binary_op.py
"""
import hashlib
import itertools
import json
import os
import stat
import sys
import ast
import csv
import configparser
from importlib import import_module
from gen_opcinfo_from_opinfo import convert_to_snake as get_op_name
from tbe.common.platform.platform_info import set_current_compile_soc_info
from impl.util.util_binary import get_bit_len


# 'pylint: disable=too-many-locals,too-many-arguments,too-many-statements
def fuzz_dtype(data_type_bit):
    dtype_dict = {
        8: 'int8',
        16: 'float16',
        32: 'float',
        64: 'int64'
    }
    return dtype_dict.get(data_type_bit)


def convert_attr_dtype(attr_type):
    """
    convert_attr_dtype, convert the attr dtype in ops json to opc input dtype
    """
    _dict = {
        "str": 'string',
        "listInt": 'list_int',
        "listFloat": 'list_float',
        "listBool": 'list_bool'
    }

    return _dict.get(attr_type, attr_type.lower())


def get_op_select_format(soc, select_format, tensor_num, attrs):
    """
    get_op_select_format, do op_select_format to get ops info
    """
    soc = "Ascend910B" if soc == "Ascend910B" else soc
    set_current_compile_soc_info(soc)
    sys.path.append(PATH.IMPL_PATH)

    try:
        dynamic_module = "impl.dynamic." + select_format
        func = getattr(import_module(dynamic_module), 'op_select_format')
    except AttributeError:
        static_module = "impl." + select_format
        func = getattr(import_module(static_module), 'op_select_format')

    input_list = [{"shape": [-2], "format": "ND", "ori_shape":[-2], "ori_format":"ND"}] * tensor_num
    if attrs is not None:
        input_list = input_list + [None] * len(attrs)
    dynamic_json = func(*input_list)
    dynamic_json = json.loads(dynamic_json)

    support_format, support_dtype = [], []
    for k in dynamic_json:
        tensor = dynamic_json.get(k)
        tensor_format = tensor.get("unknownshape_format")
        tensor_format = tensor.get("format") if tensor_format is None else tensor_format
        support_format.append(tensor_format)
        support_dtype.append(tensor.get("dtype"))
    return [support_dtype, support_format]


def get_op_supported_case(op_type, op_json, soc, select_format):
    """
    get_op_supported_case
    """
    is_format_agnostic = op_json.get("op") is not None and op_json.get("op").get("pattern") == "formatAgnostic"
    is_format_broadcast = op_json.get("op") is not None and op_json.get("op").get("pattern") == "broadcast"
    is_format_reduce = op_json.get("op") is not None and op_json.get("op").get("pattern") == "reduce"
    is_op_select = op_json.get("dynamicFormat") is not None and op_json.get("dynamicFormat").get("flag") == "true"

    input_num = 0
    output_num = 0
    for op_key in op_json.keys():
        if op_key.startswith("input"):
            input_num += 1
        elif op_key.startswith("output"):
            output_num += 1
    tensor_num = input_num + output_num

    # get attrs
    attrs = []
    if "attr" in op_json.keys():
        attr_list = op_json.get("attr").get("list").split(",")
        for i, attr_name in enumerate(attr_list):
            attr_key = "attr_" + attr_name
            attr_info = op_json.get(attr_key)
            if attr_info is None:
                print("[ERROR]{} attr is not in ops json".format(op_type))
                return None
            attr_type = attr_info.get("type")
            attr_dict = dict()
            attr_dict["name"] = attr_name
            attr_dict["dtype"] = convert_attr_dtype(attr_type)
            attrs.append(attr_dict)

    op_fixed_case = None
    if is_op_select:
        op_fixed_case = get_op_select_format(soc, select_format, tensor_num, attrs)

    # process tensor info
    def get_tensor_info(_op_key, op_fixed_case_idx):
        key_values = op_json.get(_op_key)
        # get the input dtype
        tensor_type = key_values.get("paramType")
        if tensor_type is None:
            tensor_type = "required"
        tensor_name = key_values.get("name")
        tensor_idx = int(_op_key.split("put")[1])
        if op_fixed_case is None:
            tensor_support_dtype = key_values.get("dtype").split(",")
            tensor_support_unknownshape_format = key_values.get("unknownshape_format")
            if tensor_support_unknownshape_format is None:
                if is_format_agnostic or is_format_broadcast or is_format_reduce:
                    tensor_support_unknownshape_format = ["ND"] * len(tensor_support_dtype)
                else:
                    print("[ERROR]{} is not supported unknownshape_format".format(op_type))
                    return None
            else:
                tensor_support_unknownshape_format = tensor_support_unknownshape_format.split(",")
        else:
            tensor_support_dtype = op_fixed_case[0][tensor_idx + op_fixed_case_idx].split(",")
            tensor_support_unknownshape_format = op_fixed_case[1][tensor_idx + op_fixed_case_idx].split(",")

        desc_dict = dict()
        desc_dict["name"] = tensor_name
        desc_dict["index"] = tensor_idx
        desc_dict["dtype"] = [_dtype.replace(" ", "") for _dtype in tensor_support_dtype]
        desc_dict["format"] = [_format.replace(" ", "") for _format in tensor_support_unknownshape_format]
        desc_dict["paramType"] = tensor_type

        return desc_dict

    inputs = []
    outputs = []
    for i in range(input_num):
        op_key = "input" + str(i)
        inputs_tensor = get_tensor_info(op_key, 0)
        if inputs_tensor is None:
            return None
        inputs.append(inputs_tensor)

    for i in range(output_num):
        op_key = "output" + str(i)
        outputs_tensor = get_tensor_info(op_key, input_num)
        if outputs_tensor is None:
            return None
        outputs.append(outputs_tensor)

    return inputs, outputs, attrs


def set_tensor(tensor_list):
    res_list = []
    total_case_num = len(tensor_list[0].get("dtype"))
    for i in range(total_case_num):
        case_list = []
        for _, tensor in enumerate(tensor_list):
            tmp_tensor = tensor.copy()
            if tensor.get("dtype")[i] == "float":
                tmp_tensor["dtype"] = "float32"
            else:
                tmp_tensor["dtype"] = tensor.get("dtype")[i]
            tmp_tensor["format"] = tensor.get("format")[i]
            param_type = tensor.get("paramType")
            if param_type == "optional":
                case_list.append([tmp_tensor, None])
            else:
                case_list.append([tmp_tensor])
        case_list = list(itertools.product(*case_list))
        for case in case_list:
            res_list.append(list(case))

    res_list = [json.dumps(i) for i in res_list]
    res_list = list(set(res_list))
    res_list = [json.loads(i) for i in res_list]
    return res_list


def set_attr(attr_list):
    res_list = []
    for _, attr_info in enumerate(attr_list):
        one_attr_list = []
        attr_value = attr_info.get("value")
        if attr_value is None:
            new_dict = attr_info.copy()
            new_dict["value"] = None
            one_attr_list.append(new_dict)
        else:
            for value in attr_value:
                new_dict = attr_info.copy()
                new_dict["value"] = value
                one_attr_list.append(new_dict)
        one_attr_list = [json.dumps(i) for i in one_attr_list]
        one_attr_list = list(set(one_attr_list))
        one_attr_list = [json.loads(i) for i in one_attr_list]
        res_list.append(one_attr_list)
    if res_list:
        res_list = list(itertools.product(*res_list))
    return res_list


def gen_filename_of_tensor(tensor):
    """gen_filename_of_tensor
    """
    if tensor is None:
        return "TensorNone"

    name_dype = tensor.get("dtype")
    name_shape = str(tensor.get("shape"))
    name_format = tensor.get("format")
    name_ori_shape = str(tensor.get("ori_shape", name_shape))
    name_ori_format = tensor.get("ori_format", name_format)
    name_list = [name_dype, name_shape, name_format, name_ori_shape, name_ori_format]
    name_str = "_".join(name_list)

    return name_str


class PATH:
    """
    path
    """
    OPP_PATH = os.environ.get("ASCEND_OPP_PATH", "/usr/local/Ascend/latest/opp")
    IMPL_PATH = OPP_PATH + "/op_impl/built-in/ai_core/tbe/"
    BINARY_CONFIG_PATH = "../binary_config/"


class FormatConstant:
    """
    FormatConstant
    """
    FORMAT_5HD = "NC1HWC0"
    FORMAT_6HD = "NDC1HWC0"
    FORMAT_FZ = "FRACTAL_Z"
    FORMAT_FZ_3D = "FRACTAL_Z_3D"
    FORMAT_NZ = "FRACTAL_NZ"
    FORMAT_C1HWN = "C1HWNCoC0"
    SPECIAL_FORMAT = (FORMAT_5HD, FORMAT_6HD, FORMAT_FZ, FORMAT_FZ_3D, FORMAT_NZ, FORMAT_C1HWN)
    FORMAT_ND = "ND"
    FORMAT_NCHW = "NCHW"
    FORMAT_NHWC = "NHWC"


class BinaryBase:
    """
    BinaryBase
    """
    FORMAT_MODE_DEFAULT = "FormatDefault"
    FORMAT_MODE_AGNOSTIC = "FormatAgnostic"
    FORMAT_MODE_FIXED = "FormatFixed"
    SPECIAL_FORMAT = FormatConstant.SPECIAL_FORMAT
    DTYPE_MODE_DEFAULT = "DtypeDefault"
    DTYPE_MODE_BYTE = "DtypeByte"
    FORMAT_MODE_KEY = "format_match_mode"
    DTYPE_MODE_KEY = "dtype_match_mode"
    MAGIC = "magic"
    OP_PARA_SIZE = "op_para_size"
    PARAMETERS = "parameters"
    """
    Class: class that OpBase
    """
    def __init__(self, op_type):
        self.op_type = op_type
        # when format_type is FORMAT_MODE_DEFAULT, the format not in self.special_format will be fuzz to ND
        # is op is diffent, please modify it
        self.special_format = list(BinaryBase.SPECIAL_FORMAT)
        self.input_num = 0
        self.output_num = 0
        self.binary_json = None
        self.one_binary_case_info = None
        self.is_attr_unfold = True

    def fuzz_tensor(self, tensor, format_type, dtype_type, format_nd_info=None):
        """
        do_fuzz_tensor for tensor
        """
        if format_nd_info is None:
            special_format = self.special_format
        if format_type == BinaryBase.FORMAT_MODE_AGNOSTIC:
            tensor["format"] = ["ND"] * len(tensor["format"])
        elif format_type == BinaryBase.FORMAT_MODE_DEFAULT:
            tensor["format"] = [old_format if old_format in special_format else "ND" for old_format in tensor["format"]]

        if dtype_type == BinaryBase.DTYPE_MODE_BYTE:
            tensor["dtype"] = [fuzz_dtype(get_bit_len(old_dtype)) for old_dtype in tensor["dtype"]]
        # join in shape
        tensor["shape"] = [-2]
        return tensor

    def fuzz_attr(self, attrs, attr_type):
        """
        do_fuzz_attr for attr
        """
        for i, attr_info in enumerate(attrs):
            attr_name = attr_info.get("name")
            if attr_type is not None and attr_name in attr_type.keys():
                input_attr_info = attr_type.get(attr_name)
                if input_attr_info is None:
                    attrs[i]["value"] = input_attr_info
                else:
                    attrs[i]["value"] = input_attr_info
                attr_ori_type = attr_info.get("dtype")
                attrs[i]["dtype"] = attr_ori_type
            else:
                attr_ori_type = attr_info.get("dtype")
                if attr_ori_type == "bool":
                    attrs[i]["value"] = [True, False]
                else:
                    print("[ERROR]{} attr is not bool, must input from attr_type".format(self.op_type))
                    return False
                attrs[i]["dtype"] = attr_ori_type
        return attrs

    def do_fuzz(self, op_info, format_type, dtype_type, attr_type, format_nd_info):
        """
        do_fuzz for tenser and attr
        """
        inputs, outputs, attrs = op_info
        # do fuzz tensor
        for i, tensor in enumerate(inputs):
            idx = i
            inputs[i] = self.fuzz_tensor(tensor, format_type[idx], dtype_type[idx], format_nd_info)
        for i, tensor in enumerate(outputs):
            idx = i + self.input_num
            outputs[i] = self.fuzz_tensor(tensor, format_type[idx], dtype_type[idx], format_nd_info)
        # fuzz attr
        attrs = self.fuzz_attr(attrs, attr_type)
        return [inputs, outputs, attrs]

    def transfer_json_to_list(self, json_file):
        """
        transfer binary json file to list
        """
        if not os.path.exists(json_file):
            print("[ERROR]{} the json_file is not existed".format(self.op_type))
            return False
        with open(json_file, "r") as file_op:
            ops_info_json = json.load(file_op)
        op_type = ops_info_json.get("op_type")
        if op_type is None:
            print("[ERROR]{} is not existed in file".format(self.op_type))
            return False
        if op_type != self.op_type:
            print("[ERROR]{} doesnt match".format(self.op_type))
            return False
        return ops_info_json.get("binList")

    @staticmethod
    def match_dtype(op_item, input_item):
        """
        do_match type
        """
        dtype_type = op_item.get(BinaryBase.DTYPE_MODE_KEY)
        if dtype_type == BinaryBase.DTYPE_MODE_BYTE:
            #match by dtype length
            if get_bit_len(op_item.get("dtype")) != get_bit_len(input_item.get("dtype")):
                return False
        elif dtype_type == BinaryBase.DTYPE_MODE_DEFAULT:
            if op_item.get("dtype") != input_item.get("dtype"):
                return False
        return True

    def match_format(self, op_item, input_item):
        """
        do_match format
        """
        is_legal_format = True
        format_type = op_item.get(BinaryBase.FORMAT_MODE_KEY)
        if format_type == BinaryBase.FORMAT_MODE_AGNOSTIC:
            is_legal_format = True
        elif format_type == BinaryBase.FORMAT_MODE_DEFAULT:
            special_format = self.special_format
            if op_item.get("format") == "ND" and input_item.get("format") in special_format:
                is_legal_format = False
            elif op_item.get("format") != "ND" and op_item.get("format") != input_item.get("format"):
                is_legal_format = False
        elif format_type == BinaryBase.FORMAT_MODE_FIXED:
            if op_item.get("format") != input_item.get("format"):
                is_legal_format = False
        return is_legal_format


    def is_matched(self, ops_info_file, input_tensor):
        """
        match inputs, output, attrs
        """
        op_list_json = self.transfer_json_to_list(ops_info_file)
        if op_list_json is False:
            return None
        for op_list_item in op_list_json:
            if not self.check_inputs(op_list_item, input_tensor):
                print("[error]{} check_inputs".format(self.op_type))
                continue
            if not self.check_outputs(op_list_item, input_tensor):
                print("[error]{} check_outputs".format(self.op_type))
                continue
            if not self.check_attrs(op_list_item, input_tensor):
                print("[error]{} check_attrs".format(self.op_type))
                continue
            return op_list_item
        return None

    def update_tensor(self, ops_info_file, input_tensor):
        """
        update input tensor after match

        Parameters
        ----------
        ops_info_file:json
        input_tensor:list

        Returns
        -------
        None
        """
        matched_op_item = self.is_matched(ops_info_file, input_tensor)
        if matched_op_item is None:
            print("[ERROR]{} can not find matched_op_item".format(self.op_type))
            return None
        for i, op_item in enumerate(matched_op_item.get("inputs")):
            input_tensor.get("inputs")[i]["dtype"] = op_item["dtype"]
            input_tensor.get("inputs")[i]["format"] = op_item["format"]
            input_tensor.get("inputs")[i]["ori_format"] = op_item["ori_format"]
            input_tensor.get("inputs")[i]["shape"] = tuple(op_item["shape"])
            input_tensor.get("inputs")[i]["ori_shape"] = tuple(op_item["ori_shape"])
        for i, op_item in enumerate(matched_op_item.get("outputs")):
            input_tensor.get("outputs")[i]["dtype"] = op_item["dtype"]
            input_tensor.get("outputs")[i]["format"] = op_item["format"]
            input_tensor.get("outputs")[i]["ori_format"] = op_item["ori_format"]
            input_tensor.get("outputs")[i]["shape"] = tuple(op_item["shape"])
            input_tensor.get("outputs")[i]["ori_shape"] = tuple(op_item["ori_shape"])
        if matched_op_item.get("attrs") is not None:
            for i, op_item in enumerate(matched_op_item.get("attrs")):
                if op_item["value"][0] is None:
                    input_tensor.get("attrs")[i]["value"] = None
        return True


    def check_inputs(self, op_list_item, input_tensor):
        """
        check inputs

        Parameters
        ----------
        op_list_item:list
        input_tensor:list

        Returns
        -------
        None
        """
        if len(op_list_item.get("inputs")) != len(input_tensor.get("inputs")):
            print("[ERROR]{} inputs num dosnt equal".format(self.op_type))
            return False
        inputs_num = len(op_list_item.get("inputs"))
        for i in range(inputs_num):
            op_item = op_list_item.get("inputs")[i]
            shape = op_item.get("shape")[0]
            if shape != -2:
                print("[ERROR]{} Binary shape must be -2".format(self.op_type))
                return False
            input_item = input_tensor.get("inputs")[i]
            if not self.match_format(op_item, input_item):
                print("[INFO]Binary input format doesnt match")
                return False
            if not self.match_dtype(op_item, input_item):
                print("[INFO]Binary input dtype doesnt match")
                return False
        return True

    def check_outputs(self, op_list_item, input_tensor):
        """
        check outputs

        Parameters
        ----------
        op_list_item:list
        input_tensor:list

        Returns
        -------
        None
        """
        if len(op_list_item.get("outputs")) != len(input_tensor.get("outputs")):
            print("[ERROR]{} outputs num doesnt equal".format(self.op_type))
            return False
        outputs_num = len(op_list_item.get("outputs"))
        for i in range(outputs_num):
            op_item = op_list_item.get("outputs")[i]
            shape = op_item.get("shape")[0]
            if shape != -2:
                print("[ERROR]{} Binary shape must be -2".format(self.op_type))
                return False
            output_item = input_tensor.get("outputs")[i]
            if not self.match_format(op_item, output_item):
                print("[INFO]Binary output format doesnt match")
                return False
            if not self.match_dtype(op_item, output_item):
                print("[INFO]Binary output dtype doesnt match")
                return False
        return True

    def check_attrs(self, op_list_item, input_tensor):
        """
        check attrs

        Parameters
        ----------
        op_list_item:list
        input_tensor:list

        Returns
        -------
        None
        """
        op_attrs = op_list_item.get("attrs")
        if op_attrs is None or op_attrs == []:
            print("[INFO]there is no attr attribute for this op")
            return True
        input_attrs = input_tensor.get("attrs")
        if input_attrs is None:
            return False
        for _, op_attr in enumerate(op_attrs):
            op_attr_name = op_attr.get("name")
            input_attr_item = self.find_attr(op_attr_name, input_attrs)
            if input_attr_item is None:
                return False
            if op_attr.get("value")[0] is None:
                continue
            if input_attr_item.get("value") not in op_attr.get("value"):
                print("[ERROR]{} attr value doesnt equal".format(self.op_type))
                return False
        return True

    def find_attr(self, op_attr_name, input_attrs):
        """
        find attr by attr name

        Parameters
        ----------
        op_list_item:list
        input_attrs:list

        Returns
        -------
        None
        """
        for input_attr_item in input_attrs:
            if input_attr_item.get("name") == op_attr_name:
                return input_attr_item
        print("[ERROR]{} attr name doesnt match".format(self.op_type))
        return None

    def check_ops_info(self, ops_info_file):
        """
        check_ops_info
        """
        if not os.path.exists(ops_info_file):
            print("[ERROR]{} ops_info_file is not existed".format(self.op_type))
            return False

        with open(ops_info_file, "r") as file_op:
            ops_info_json = json.load(file_op)

        op_json = ops_info_json.get(self.op_type)
        if op_json is None:
            print("[ERROR]{} is not existed in {}".format(self.op_type, ops_info_file))
            return False
        return op_json

    def add_gen_binary(self, result, format_type, dtype_type):
        for k, case in enumerate(result):
            tensor_list, attr_list = case
            bin_filename = []
            for i, tensor in enumerate(tensor_list):
                if tensor is not None:
                    tensor_format_mode, tensor_dtype_mode = format_type[i], dtype_type[i]
                    # add special key to json for match
                    tensor[BinaryBase.FORMAT_MODE_KEY] = tensor_format_mode
                    tensor[BinaryBase.DTYPE_MODE_KEY] = tensor_dtype_mode
                    if tensor_dtype_mode != BinaryBase.DTYPE_MODE_DEFAULT:
                        tensor[BinaryBase.DTYPE_MODE_KEY] = tensor_dtype_mode
                    # if the format is ND, change the FORMAT_MODE_KEY to BinaryBase.FORMAT_MODE_DEFAULT
                    support_mod = (tensor[BinaryBase.FORMAT_MODE_KEY] != BinaryBase.FORMAT_MODE_AGNOSTIC)
                    if tensor["format"] == "ND" and support_mod:
                        tensor[BinaryBase.FORMAT_MODE_KEY] = BinaryBase.FORMAT_MODE_DEFAULT
                    if tensor_format_mode == BinaryBase.FORMAT_MODE_DEFAULT:
                        tensor.pop(BinaryBase.FORMAT_MODE_KEY)
                    if tensor_dtype_mode == BinaryBase.DTYPE_MODE_DEFAULT:
                        tensor.pop(BinaryBase.DTYPE_MODE_KEY)

                if i < self.input_num:
                    bin_filename = self.add_input_tensor(tensor, bin_filename)
                else:
                    bin_filename = self.add_output_tensor(tensor, bin_filename)
            bin_filename = self.add_attr(attr_list, bin_filename)
            self.add_bin_file_name(bin_filename)
            self.add_binary_case()

    def gen_binary_json(self, ops_info_file, format_type, dtype_type, attr_type, format_nd_info, soc, select_format):
        """
        gen_binary_json

        Parameters
        ----------
        ops_info_file : string
            the ops info path
        format_type: list or None
            the format rule key for tensor must in [FORMAT_MODE_DEFAULT, FORMAT_MODE_AGNOSTIC, FORMAT_MODE_FIXED]
        dtype_type: list or None
            the dtype rule key for tensor must in [DTYPE_MODE_DEFAULT, DTYPE_MODE_BYTE]
        attr_type: list or None
            dict info, the key is attr name, the value is list or None
        format_nd_info: dict
        op_fixed_case: list

        Returns
        -------
        None
        """
        # get supported case from ops_json
        op_json = self.check_ops_info(ops_info_file)
        # fuzz the supported case
        if not op_json:
            return False
        ana_res = get_op_supported_case(self.op_type, op_json, soc, select_format)
        if ana_res is None:
            print("[ERROR]{} get_op_supported_case failed".format(self.op_type))
            return False
        self.input_num, self.output_num = len(ana_res[0]), len(ana_res[1])

        if len(format_type) == 1:
            format_type = format_type * (self.input_num + self.output_num)
        elif not isinstance(format_type, list):
            format_type = [format_type] * (self.input_num + self.output_num)
        format_type = [BinaryBase.FORMAT_MODE_DEFAULT if key is None else key for key in format_type]

        if len(dtype_type) == 1:
            dtype_type = dtype_type * (self.input_num + self.output_num)
        elif not isinstance(dtype_type, list):
            dtype_type = [dtype_type] * (self.input_num + self.output_num)
        dtype_type = [BinaryBase.DTYPE_MODE_DEFAULT if key is None else key for key in dtype_type]
        # do fuzz
        ana_res = self.do_fuzz(ana_res, format_type, dtype_type, attr_type, format_nd_info)
        inputs, outputs, attrs = ana_res

        # do set for binary case
        binary_tensor_list = set_tensor(inputs + outputs)
        if self.is_attr_unfold:
            binary_attr_list = set_attr(attrs)
            binary_attr_list = [[]] if not binary_attr_list else binary_attr_list
        else:
            binary_attr_list = [attrs.copy()]

        result = list(itertools.product(binary_tensor_list, binary_attr_list))

        # gen binary json
        self.add_gen_binary(result, format_type, dtype_type)
        return True

    def add_bin_file_name(self, bin_filename):
        """
        add_bin_filename to one_binary_case_info
        """
        if self.one_binary_case_info is None:
            self.init_binary_case()
        bin_filename = "_".join(bin_filename)
        self.one_binary_case_info["bin_filename"] = bin_filename

    def init_binary_json(self):
        """
        init_binary_json
        """
        self.binary_json = dict()
        self.binary_json["op_type"] = self.op_type
        self.binary_json["op_list"] = []

    def init_binary_case(self):
        """
        init_binary_case
        """
        self.one_binary_case_info = dict()
        self.one_binary_case_info["bin_filename"] = ""
        self.one_binary_case_info["inputs"] = []
        self.one_binary_case_info["outputs"] = []

    def add_binary_case(self):
        """
        add_binary_case
        """
        if self.binary_json is None:
            self.init_binary_json()
        self.binary_json.get("op_list").append(self.one_binary_case_info)
        self.one_binary_case_info = None

    def add_input_tensor(self, tensor, bin_filename):
        """
        add_input_tensor

        Parameters
        ----------
        tensor : dict
            tensor info

        Returns
        -------
        None
        """
        if self.one_binary_case_info is None:
            self.init_binary_case()

        self.one_binary_case_info.get("inputs").append(tensor)

        bin_filename.append(gen_filename_of_tensor(tensor))

        return bin_filename

    def get_input_tensor(self, tensor_key):
        """
        get_input_tensor base on tensor_key
            when tensor_key is int, use tensor idx
            when tensor_key is str, use tensor name to get tensor
        """
        if isinstance(tensor_key, int):
            return self.one_binary_case_info.get("inputs")[tensor_key]
        tensor_name = str(tensor_key)
        for tensor in self.one_binary_case_info.get("inputs"):
            if tensor.get("name") == tensor_name:
                return tensor
        return None

    def add_output_tensor(self, tensor, bin_filename):
        """
        add_output_tensor

        Parameters
        ----------
        tensor : dict
            tensor info

        Returns
        -------
        None
        """
        if self.one_binary_case_info is None:
            self.init_binary_case()

        self.one_binary_case_info.get("outputs").append(tensor)

        bin_filename.append(gen_filename_of_tensor(tensor))

        return bin_filename

    def get_output_tensor(self, tensor_key):
        """
        get_output_tensor base on tensor_key
            when tensor_key is int, use tensor idx
            when tensor_key is str, use tensor name to get tensor
        """
        if isinstance(tensor_key, int):
            return self.one_binary_case_info.get("outputs")[tensor_key]
        tensor_name = str(tensor_key)
        for tensor in self.one_binary_case_info.get("outputs"):
            if tensor.get("name") == tensor_name:
                return tensor
        return None

    def add_attr(self, attr_list, bin_filename):
        """
        add attr list to one_binary_case_info
        """
        if self.one_binary_case_info is None:
            self.init_binary_case()
        if attr_list != []:
            for attr in attr_list:
                bin_filename.append(str(attr.get("value")))
            self.one_binary_case_info["attrs"] = attr_list
        return bin_filename

    def get_attr(self, attr_key):
        """
        get_attr base on attr_key
            when attr_key is int, use attr idx to get attr
            when attr_key is str, use attr name to get attr
        """
        if isinstance(attr_key, int):
            return self.one_binary_case_info.get("attrs")[attr_key]
        attr_name = str(attr_key)
        for attr_info in self.one_binary_case_info.get("attrs"):
            if attr_info["name"] == attr_name:
                return attr_info
        return None

    def dump_binary_json_to_file(self, file_path):
        """
        dump_binary_json_to_file
        """
        flags = os.O_WRONLY | os.O_CREAT
        modes = stat.S_IWUSR | stat.S_IRUSR
        with os.fdopen(os.open(file_path, flags, modes), 'w') as fp:
            fp.truncate()
            self.binary_json.get("op_list").sort(key=lambda x: x.get("bin_filename"))
            for i, kernel in enumerate(self.binary_json.get("op_list")):
                bin_filename = kernel.get("bin_filename")
                hash_md5 = hashlib.md5()
                hash_md5.update(bin_filename.encode())
                hash_bin_filename = hash_md5.hexdigest()
                op_bin_filename = self.op_type + "_" + str(hash_bin_filename)
                self.binary_json.get("op_list")[i]["bin_filename"] = op_bin_filename
            json.dump(self.binary_json, fp, indent=2)

    def load_binary_file_to_json(self, file_path):
        """
        load_binary_file_to_json
        """
        if not os.path.exists(file_path):
            print("[ERROR]{} binary_file is not existed".format(self.op_type))
            self.binary_json = None
        with open(file_path, "r") as file_op:
            self.binary_json = json.load(file_op)


#---------------------------------------------------------------------------------
#  binary_fuzz_json
def writer_config_csv(op_type, op_name, config_csv_path):
    """
    writer_config_csv
    """
    with open (config_csv_path) as f:
        read_csv = csv.reader(f)
        column = [row[0] for row in read_csv]
    if op_type not in column:
        flags = os.O_WRONLY | os.O_CREAT
        modes = stat.S_IWUSR | stat.S_IRUSR
        with os.fdopen(os.open(config_csv_path, flags, modes), 'a') as f:
            row = [op_type, op_name + ".json"]
            write_csv = csv.writer(f)
            write_csv.writerow(row)


def get_operator_func(csv_path):
    """
    get_operator_func
    """
    with open (csv_path) as f:
        f_csv = csv.DictReader(f)
        operator_func = {}
        for item in f_csv:
            func = item.get("file_name")
            func = func.replace(".py", "")
            func = func.replace("dynamic/", "")
            operator_func[item.get("op_type")] = func
    return operator_func


def fuzz_opinfo_cfg(op_type, op_name, soc_version, format_type, dtype_type, attrs, nd_info):
    """
    fuzz_opinfo_cfg
    """
    for soc in soc_version:
        op_ob = BinaryBase(op_type)
        cfg_soc = soc.lower()
        op_info_cfg = "./aic-" + cfg_soc + "-ops-info.json"
        csv_path = "./aic-" + cfg_soc + "-opc-info.csv"
        op_info_binary_cfg_path = PATH.BINARY_CONFIG_PATH + cfg_soc + "/" + op_type + "/"
        config_csv_path =  PATH.BINARY_CONFIG_PATH + "binary_config.csv"

        os.system("bash gen_opinfo_json_from_ini.sh {} {}".format(cfg_soc, op_info_cfg))
        os.system("python3 gen_opcinfo_from_opinfo.py {} {}".format(op_info_cfg, csv_path))
        operator_func = get_operator_func(csv_path)
        select_format = operator_func.get(op_type)

        is_gen = op_ob.gen_binary_json(op_info_cfg, format_type, dtype_type,
                                       attrs, nd_info, soc, select_format)
        if is_gen:
            try:
                os.makedirs(op_info_binary_cfg_path)
                op_info_binary_cfg = op_info_binary_cfg_path + op_name + ".json"
                op_ob.dump_binary_json_to_file(op_info_binary_cfg)
                print("[INFO] success, {} dump binary json in {}".format(op_type, soc))
            except FileExistsError:
                print("[Warning]{} binary config already exists in {}".format(op_type, soc))
                op_info_binary_cfg = op_info_binary_cfg_path + op_name + ".json"
                op_ob.dump_binary_json_to_file(op_info_binary_cfg)
                print("[INFO] success, {} dump binary json in {}".format(op_type, soc))
            writer_config_csv(op_type, op_name, config_csv_path)
        else:
            print("[ERROR]{} dump binary json fail in {}".format(op_type, soc))


def get_attrs(var_attrs, enumerate_attrs):
    """
    get_attrs
    """
    attrs = {}
    if var_attrs is not None:
        var_attrs = var_attrs.strip(',').split(',')
        values = [None] * len(var_attrs)
        var_attrs = dict(zip(var_attrs, values))
        attrs.update(var_attrs)
    if enumerate_attrs is not None:
        enumerate_attrs = ast.literal_eval(enumerate_attrs)
        attrs.update(enumerate_attrs)
    return attrs


def binary_cfg(op_type, soc_version):
    """
    read binary_json_cfg
    """

    # mode dict
    format_mode = {"DEFAULT": BinaryBase.FORMAT_MODE_DEFAULT, "AGNOSTIC": BinaryBase.FORMAT_MODE_AGNOSTIC,
                   "FIXED": BinaryBase.FORMAT_MODE_FIXED}
    dtype_mode = {"DEFAULT": BinaryBase.DTYPE_MODE_DEFAULT, "BYTE": BinaryBase.DTYPE_MODE_BYTE}

    # 读取ini文件
    cfg = configparser.ConfigParser()
    cfg.read("binary_json_cfg.ini")

    # 获取需要生成json的算子
    if op_type == "all":
        op_type = cfg.sections()
    else:
        op_type = op_type.strip(',').split(',')

    # 获取需要生成json的平台
    if soc_version == "all":
        soc_version = ["Ascend310", "Ascend310B", "Ascend610", "Ascend615", "Ascend310P",
                       "Ascend910", "Ascend910B", "Hi3796CV300ES", "Hi3796CV300CS", "SD3403"]
    else:
        soc_version = soc_version.strip(',').split(',')

    # 生成算子的json
    for operator in op_type:
        try:
            items = dict(cfg.items(operator))
        except configparser.NoSectionError:
            sys.exit("[ERROR]{} not in binary json config".format(operator))

        # get format_type
        format_type = items.get("format_type")
        if format_type is None:
            format_type = ["DEFAULT"]
        else:
            format_type = format_type.strip(',').split(',')
        for i, dtype in enumerate(format_type):
            format_type[i] = format_mode.get(dtype)

        # get dtype_type
        dtype_type = items.get("dtype_type")
        if dtype_type is None:
            dtype_type = ["DEFAULT"]
        else:
            dtype_type = dtype_type.strip(',').split(',')
        for i, dtype in enumerate(dtype_type):
            dtype_type[i] = dtype_mode.get(dtype)

        # get optional parameter
        op_name = get_op_name(operator)
        nd_info = items.get("nd_info")
        var_attrs = items.get("var_attrs")
        enumerate_attrs = items.get("enumerate_attrs")
        attrs = get_attrs(var_attrs, enumerate_attrs)
        attrs = None if attrs == {} else attrs

        # fuzz json
        fuzz_opinfo_cfg(operator, op_name, soc_version, format_type, dtype_type, attrs, nd_info)


def mate_json(op_type, binary_file, input_tensors):
    op_ob = BinaryBase(op_type)
    with open(input_tensors, 'r') as tensours:
        input_tensour = json.load(tensours)
    op_ob.update_tensor(binary_file, input_tensour)
