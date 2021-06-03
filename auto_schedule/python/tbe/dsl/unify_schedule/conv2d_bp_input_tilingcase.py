# Copyright 2019-2020 Huawei Technologies Co., Ltd
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
conv2d backprop input tiling case
"""
import copy
import json
from collections import OrderedDict
from functools import reduce

from tbe.common.context import get_context
from tbe.common.platform import platform_info as tbe_platform_info
from tbe.common.tiling.get_tiling import get_tiling
from tbe.dsl.base.operation import get_te_var
from tbe.dsl.base.operation import register_tiling_case
from tbe.dsl.base.operation import register_build_pointcut
from tbe.dsl.base.operation import get_context as op_get_context
from tbe.dsl.base.operation import add_compile_info
from tbe.dsl.base.operation import get_compile_info
from tbe.dsl.compute.conv2d_backprop_input_compute import DynamicConv2dBpInputParams
from tbe.tvm.expr import Expr

from .cube_tilingcase import TilingSelection
from .cube_tilingcase import CubeTilingOp
from .cube_tilingcase import TilingUtils as utils
from .constants import Pattern
from .tilingcase_util import Conv2dBackpropParaProcess


W_DELTA = 1
H_LEN = 400
W_LEN = 400
LARGE_M = 10000
MIN_STEP = 16
FUSED_DOUBLE_OPERAND_MUL = 100.0
BIT_RATIO_DICT = {"int32": 4, "float32": 4, "float16": 2,
                  "uint8": 1, "int8": 1, "uint4": 0.5, "int4": 0.5}
DEFAULT_KERNEL_ID = None


def parse_fuzz_build_range(info_list):
    """
    parse multiple range segment from json string

    Notice
    ----------
    for conv2d backprop input only parse input range

    Parameters
    ----------
    info_list: list support info
        [{
            "inputs": [{
                "index": 0,
                "tensor": [{
                    "range": [
                        [16, 32],
                        [3, 3],
                        [16, 32],
                        [16, 32]
                    ],
                    "shape": [-1, 3, -1, -1]
                }]
            }]
        }]

    Returns
    -------
    range_list: list of 4d range
    """
    invalid = (not isinstance(info_list, list)) or len(info_list) == 0
    if invalid:
        raise RuntimeError("invalid missing support info {}".format(str(info_list)))
    list_size = 4
    range_list = []
    op_type = DynamicConv2dBpInputParams.dynamic_para.get("op_type")
    if op_type == "Conv2DBackpropInput" or op_type == "depthwise_conv2d_backprop_input":
        target_index = 2
    elif op_type == "Conv2DTranspose":
        target_index = 1
    else:
        target_index = 0
    for item in info_list:
        inputs = item.get("inputs")
        invalid = (not isinstance(inputs, list)) or len(inputs) == 0
        if invalid:
            continue
        # >>> start: parse range from index [0] input
        for input_tensor in inputs:
            invalid = (not isinstance(input_tensor, dict)) or input_tensor.get("index") != target_index
            if invalid:
                continue
            invalid = (not isinstance(input_tensor.get("tensor"), list)) \
                      or len(input_tensor.get("tensor")) == 0 \
                      or (not isinstance(input_tensor.get("tensor")[0].get("range"), list)) \
                      or len(input_tensor.get("tensor")[0].get("range")) != list_size
            if invalid:
                raise RuntimeError("invalid support info input {}".format(str(input_tensor)))
            input_range = input_tensor.get("tensor")[0].get("range")
            for axis_range in input_range:
                invalid = (not isinstance(axis_range, list)) \
                          or len(axis_range) != 2 \
                          or axis_range[0] < 1 \
                          or axis_range[0] > axis_range[1]
                if invalid:
                    raise RuntimeError("invalid range {}".format(str(axis_range)))
            range_list.append(input_range)
            # <<< end: parse range from index [0] input
    return range_list


def gen_support_info(range_x, ori_tensors):
    """
    kernel list support info part

    Notice
    ------
    only need to set inputs with range

    Parameters
    ----------
    range_x: list
         input x range
    ori_tensors: dict
        orginal vaild tensors

    Returns
    -------
    support_info: dict
    """
    support_info = {}
    # >>> start: generate input shape and range
    inputs = []
    item = {}
    op_type = DynamicConv2dBpInputParams.dynamic_para.get("op_type")
    if op_type == "Conv2DBackpropInput" or op_type == "depthwise_conv2d_backprop_input":
        item["index"] = 2
    elif op_type == "Conv2DTranspose":
        item["index"] = 1
    else:
        item["index"] = 0
    item["tensor"] = []
    tensor_info = {}
    if op_type == "Conv2DBackpropInput" or op_type == "depthwise_conv2d_backprop_input":
        ori_tensors_input = ori_tensors.get("out_backprop")
    else:
        ori_tensors_input = ori_tensors.get("x")
    ori_shape = ori_tensors_input.get("ori_shape")
    tensor_info["shape"] = ori_shape
    x_format = ori_tensors_input.get("ori_format")

    # get dy_range depends on dx_range
    conv_info = DynamicConv2dBpInputParams.tiling_info_dict
    dx_range = [range_x[0], [ori_shape[x_format.find("C")], ori_shape[x_format.find("C")]], range_x[1], range_x[2]]
    data_format = x_format
    ori_paras = {
        "filters": ori_tensors.get("filters"), "bias": None, "offset_w": None,
        "strides": (conv_info.get("strideH"), conv_info.get("strideW"), conv_info.get("strideH_expand"),
                    conv_info.get("strideW_expand")),
        "pads": (conv_info.get("padu"), conv_info.get("padd"), conv_info.get("padl"), conv_info.get("padr")),
        "dilations": (1, 1, 1, 1), "groups": conv_info.get("group"), "data_format": data_format,
        "output_padding": (0, 0, 0, 0), "offset_x": 0, "kernel_name": conv_info.get("kernel_name")
    }
    conv2d_backprop = Conv2dBackpropParaProcess(ori_paras)
    filter_shape_nchw = conv2d_backprop.get_input_nchw(ori_tensors.get("filters").get("ori_shape"),
                                                       ori_tensors.get("filters").get("ori_format"))
    dx_range_nchw = dx_range
    dy_range_nchw, _, _ = conv2d_backprop.get_output_range(filter_shape_nchw, dx_range_nchw)
    if op_type == "depthwise_conv2d_backprop_input":
        dy_range_nchw[1] = [filter_shape_nchw[0] * filter_shape_nchw[1], filter_shape_nchw[0] * filter_shape_nchw[1]]
    range_valid = [[0, 0]] * 4
    range_valid[x_format.find("N")] = list(dy_range_nchw[0])
    range_valid[x_format.find("C")] = list(dy_range_nchw[1])
    range_valid[x_format.find("H")] = list(dy_range_nchw[2])
    range_valid[x_format.find("W")] = list(dy_range_nchw[3])
    tensor_info["range"] = range_valid
    item["tensor"].append(tensor_info)
    inputs.append(item)
    support_info["inputs"] = inputs
    # <<< end: generate input shape and range
    return support_info


def add_covered_shape_range(compile_info):
    """
    tiling_case func for dynamic shape conv2d backprop input

    Parameters
    ----------
    compile_info: dict
        tiling range info

    Returns
    -------
    info_list: dict
        support info and compile info pair
    max_id: int
        last kernel id
    """
    info_list = []
    id_list = list(compile_info["block_dim"].keys())
    id_list.sort()
    max_id = id_list[-1]
    # >>> start: add compute var for op tiling
    te_vars = []
    for cpt in op_get_context().get_computes():
        te_vars += cpt.get_vars()
    var_list = [var.get_name() for var in te_vars]

    # <<< end: add compute var for op tiling
    for kernel_id, block_id in compile_info["block_dim"].items():
        new_compile = compile_info.copy()
        # >>> start: keep only one record
        for keys, value in new_compile.items():
            if isinstance(value, dict):
                value = {} if value.get(kernel_id) is None else {kernel_id: value[kernel_id]}
                new_compile[keys] = value
        # <<< end: keep only one record
        new_compile["kernelId"] = kernel_id
        new_compile["_vars"] = {kernel_id: var_list}
        range_x = new_compile["repo_range"].get(kernel_id) or new_compile["cost_range"].get(kernel_id)
        new_range = [range_x[:2], range_x[2:4], range_x[4:6]]
        ori_tensors = DynamicConv2dBpInputParams.ori_tensor
        new_support = gen_support_info(new_range, ori_tensors)
        info_list.append({"supportInfo": new_support, "compileInfo": new_compile})
    return info_list, max_id


@register_build_pointcut(pattern=Pattern.CONV2D_BACKPROP_INPUT)
def build_pointcut_conv2d_backprop_input(func, *args, **kwargs):
    """
    kernel info process before build

    Notice
    ------
    kernel_info: dict with support info and compile info
        {
            "supportInfo": {
                "inputs": [{
                    "index": 0,
                    "tensor": [{
                        "range": [
                            [16, 32],
                            [3, 3],
                            [64, 128],
                            [64, 128]
                        ],
                        "shape": [-1, 3, -1, -1]
                    }]
                }]
            },
            "compileInfo": {
                "_pattern": "Conv2d_backprop_input",
                "tiling_type": "dynamic_tiling",
                "repo_seeds": {},
                "repo_range": {},
                "cost_range": {
                    1: [16, 32, 64, 128, 64, 128]
                },
                "block_dim": {
                    1: 16
                },
                "_vars": {
                    1: ["batch_n", "dedy_h", "dx_h", "dedy_w", "dx_w"]
                }
            }
        }

    Parameters
    ----------
    func: funtions
        build process
    args: list
        function input args
    kwargs: dict
        function input args and value

    Returns
    -------
    None
    """
    fuzz_build = get_context().get_build_type() == "fuzzily_build"
    if fuzz_build:  # set kernel info
        info_list, max_id = add_covered_shape_range(get_compile_info())
        get_context().add_build_json_result("kernelList", info_list)
        get_context().add_build_json_result("maxKernelId", max_id)
    func(*args, **kwargs)


@register_tiling_case(pattern=Pattern.CONV2D_BACKPROP_INPUT)
def calc_conv2dbp_input(outs, option=None):
    """
    tiling_case func for dynamic shape conv2d_bp_input

    Parameters
    ----------
    outs : tvm tensor or list of tvm tensor, results for tvm compute

    Returns
    -------
    list of dict, each dict for a tiling case
    """

    var_names = ("batch_n", "dx_h", "dx_w")
    fuzz_build = get_context().get_build_type() == "fuzzily_build"
    conv_info = DynamicConv2dBpInputParams.tiling_info_dict
    if outs[-1].op.tag == "elewise_multiple_sel":
        conv_info["fused_double_operand_num"] = 1 / 16
        if "elewise_binary_add" in outs[-1].op.input_tensors[1].op.tag:
            conv_info["fused_double_operand_num"] += 1
            conv_info["fusion_type"] = 4
    tgt_list = []
    tgt_area = {}
    shape_dict = {"batch_n": conv_info.get("C_shape")[0],
                  "dx_h": conv_info.get("C_shape")[2],
                  "dx_w": conv_info.get("C_shape")[3]}

    for var_name in var_names:
        if get_te_var(var_name):
            tgt_area[var_name] = tuple(get_te_var(var_name).get_bound())
        else:
            tgt_area[var_name] = (int(shape_dict.get(var_name)), int(shape_dict.get(var_name)))
    tgt_list.append(tgt_area)
    max_id = DEFAULT_KERNEL_ID
    if fuzz_build:  # parse input range
        # generate tgt_area by format
        ori_tensors = DynamicConv2dBpInputParams.ori_tensor
        op_type = DynamicConv2dBpInputParams.dynamic_para.get("op_type")
        if op_type == "Conv2DBackpropInput" or op_type == "depthwise_conv2d_backprop_input":
            ori_tensors_input = ori_tensors.get("out_backprop")
        else:
            ori_tensors_input = ori_tensors.get("x")
        invalid = (not isinstance(ori_tensors, dict)) \
                  or (not isinstance(ori_tensors_input, dict))
        if invalid:
            raise RuntimeError("can't get input from para_dict")
        input_format = ori_tensors_input["ori_format"]
        pos_list = [input_format.find("N"),
                    input_format.find("H"),
                    input_format.find("W")]
        # te fusion make sure that each range is within the range request
        range_str = get_context().get_addition("missing_support_info")
        range_list = []
        if len(range_str) > 0:
            range_list = parse_fuzz_build_range(json.loads(range_str))
        if len(range_list) > 0:
            tgt_list.clear()
            for item in range_list:
                # get dx_range depends on dy_range
                dy_range = copy.deepcopy(item)
                data_format = input_format
                ori_paras = {
                    "filters": ori_tensors.get("filters"), "bias": None, "offset_w": None,
                    "strides": (conv_info.get("strideH"), conv_info.get("strideW"), conv_info.get("strideH_expand"),
                                conv_info.get("strideW_expand")),
                    "pads": (
                    conv_info.get("padu"), conv_info.get("padd"), conv_info.get("padl"), conv_info.get("padr")),
                    "dilations": (1, 1, 1, 1), "groups": conv_info.get("group"), "data_format": data_format,
                    "output_padding": (0, 0, 0, 0), "offset_x": 0, "kernel_name": conv_info.get("kernel_name")
                }
                conv2d_backprop = Conv2dBackpropParaProcess(ori_paras)
                filter_shape_nchw = conv2d_backprop.get_input_nchw(ori_tensors.get("filters").get("ori_shape"),
                                                                   ori_tensors.get("filters").get("ori_format"))
                _, dy_range_nchw = conv2d_backprop.get_input_nchw((1, 1, 1, 1), data_format, dy_range)
                dx_range_nchw, _, _ = conv2d_backprop.get_input_range(filter_shape_nchw, dy_range_nchw)
                fuzz_area = {}
                fuzz_area["batch_n"] = tuple(dx_range_nchw[0])
                fuzz_area["dx_h"] = tuple(dx_range_nchw[2])
                fuzz_area["dx_w"] = tuple(dx_range_nchw[3])
                tgt_list.append(fuzz_area)
        # >>> start: get kernel id
        kernel_id = get_context().get_addition("max_kernel_id")
        valid = isinstance(kernel_id, int) and kernel_id > -2
        if valid:
            max_id = kernel_id + 1

    tiling_cases = []
    total_info = {}
    for tgt in tgt_list:
        new_info = copy.deepcopy(conv_info)
        tiling_op = Conv2dBpInputTiling(new_info, DynamicConv2dBpInputParams.var_map)
        selector = TilingSelection(tiling_op, max_id)
        tiling_cases += selector.calc_tiling(tgt, var_names)
        # >>> start: gather compile_info process
        if fuzz_build:
            tgt_nhw = []
            for var_name in var_names:
                tgt_nhw.extend(tgt[var_name])
            current_info = get_compile_info().copy()
            id_list = list(current_info["block_dim"].keys())
            id_list.sort()
            max_id = id_list[-1] + 1
            # >>> start: make sure range is within tgt_nhw
            for range_key in ["repo_range", "cost_range"]:
                valid = isinstance(current_info.get(range_key), dict)
                if valid:
                    for kernel_id, range_x in current_info[range_key].items():
                        new_range = []
                        for index, dim_value in enumerate(range_x):
                            if index in (0, 2, 4):
                                new_range.append(tgt_nhw[index] if dim_value < tgt_nhw[index] else dim_value)
                            else:
                                new_range.append(tgt_nhw[index] if dim_value > tgt_nhw[index] else dim_value)
                        current_info[range_key][kernel_id] = new_range
            # <<< end: make sure range is within tgt_nhw
            if total_info:
                # >>> start: add new dict info
                for key, value in current_info.items():
                    need_update = isinstance(total_info.get(key), dict) \
                                  and isinstance(value, dict)
                    if need_update:
                        new_item = total_info[key]
                        new_item.update(value)
                        total_info[key] = new_item
                        add_compile_info(key, total_info[key])
            else:
                total_info = current_info
                # <<< end: add new dict info
        # <<< end: gather compile_info process
    return tiling_cases


class Conv2dBpInputTiling(CubeTilingOp):
    """
    get_tiling class for dynamic shape conv2d_bp_input
    """
    def __init__(self, tiling_info, var_map):
        super().__init__(tiling_info, None, var_map)
        self.a_info = self.tiling_info['A_shape']
        self.b_info = self.tiling_info['B_shape']
        self.c_info = self.tiling_info['C_shape']
        self.a_type = self.tiling_info["A_dtype"]
        self.c_type = self.tiling_info["C_dtype"]
        self.var_map = var_map
        self._get_calc_info()
        self.key = 'C_shape'
        self.op_type = "conv2d_bp_input"

    def _modify_repo_tiling(self, tiling_mess):
        tiling = tiling_mess.get("tiling")
        n_size = tiling_mess.get("B_shape")[1]
        block_dim = tiling.get("block_dim")
        nc = tiling.get("CL0_matrix")[0]
        n_factor = utils.icd(n_size // block_dim[1], nc)
        block_dim[1] = n_size // nc // n_factor

    def get_repo_tiling(self):
        """
        get tiling from repository

        Returns
        -------
        tiling: shape and tiling retrieved from repository
        """

        tiling_list = get_tiling(self.tiling_info)
        res_list = []
        for tiling_mess in tiling_list:
            self._modify_repo_tiling(tiling_mess)
            # in dx_opti, tiling's C_shape returned from repository is 0,
            # we calculate C_shape according to A_shape and stride
            if tiling_mess["C_shape"][0] == -1:
                tiling_mess["C_shape"][0] = tiling_mess["A_shape"][0]
            if tiling_mess["C_shape"][2] == 0:
                tiling_mess["C_shape"][2] = tiling_mess["A_shape"][2] * self.stride_h
            if tiling_mess["C_shape"][3] == 0:
                tiling_mess["C_shape"][3] = tiling_mess["A_shape"][3] * self.stride_w
            # pad set -1 to get tilings from repository, so we need to
            # check A_shape&C_shape to filter tilings not matched with
            # current kernel_info out
            t_h, t_w = self.get_output_h(tiling_mess["C_shape"][2]), \
                self.get_output_w(tiling_mess["C_shape"][3])
            if (t_h == tiling_mess["A_shape"][2] and t_w == tiling_mess["A_shape"][3]
                and self.check_tiling_ub(tiling_mess)):
                res_list.append(tiling_mess)
        return res_list

    def check_tiling_ub(self, tiling_mess):
        """
        Check if tiling in repository ub space is legal

        Parameters
        ----------
        tiling_mess: shape and tiling retrieved from repository

        Returns
        -------
        tiling_valid_flag: If true means it's legal
        """
        tiling = tiling_mess.get('tiling')
        cub_db_flag = tiling.get("manual_pingpong_buffer").get("CUB_pbuffer")
        cub_dtype_bit = BIT_RATIO_DICT.get(self.c_type)
        cub_size = (reduce(lambda x, y: x * y, tiling.get("CUB_matrix")) * cub_db_flag * cub_dtype_bit)
        fused_double_operand_num = tiling_mess.get('fused_double_operand_num')
        fused_double_operand_num = fused_double_operand_num if fused_double_operand_num is not None else 0
        fused_double_operand_num /= FUSED_DOUBLE_OPERAND_MUL
        ub_size_limit = tbe_platform_info.get_soc_spec("UB_SIZE")
        bias_flag = 1 if tiling_mess.get('bias_flag') == True else 0
        bias_size = tiling_mess.get("C_shape")[1] * tiling_mess.get("C_shape")[-1] * bias_flag * cub_dtype_bit
        if (self.stride_h > 1 or self.stride_w > 1):
            if tiling.get("AUB_shape"):
                aub_tiling_k, aub_tiling_m, _, _ = tiling.get("AUB_shape")
                aub_co1 = aub_tiling_k // (self.b_info[2] * self.b_info[3] * utils.FP16_K)
                aub_w = tiling_mess["A_shape"][3] * self.stride_w
                aub_h = (aub_tiling_m + self.stride_h - 1) // self.stride_h
                aub_db = tiling.get("manual_pingpong_buffer").get("AUB_pbuffer")
                aub_bit = BIT_RATIO_DICT.get(self.a_type)
                aub_filling_size = aub_co1 * aub_tiling_m * aub_w * utils.FP16_K * aub_db * aub_bit
                cub_size *= (1 + fused_double_operand_num)
                if (cub_size + aub_filling_size + bias_size) > ub_size_limit:
                    return False
            elif self.k_h == 1 and self.k_w == 1:
                dedy_h, dedy_w = tiling_mess.get("A_shape")[2:4]
                dx_w = tiling_mess.get("C_shape")[3]
                nc_factor, mc_factor, m0, n0 = tiling.get("CUB_matrix")[:4]
                mc_from_tiling = mc_factor * m0
                max_n_is_hfactor = (ub_size_limit - cub_size) // (
                    nc_factor * n0 * cub_db_flag * cub_dtype_bit * self.stride_h) // dx_w
                if mc_from_tiling >= dedy_w:
                    if mc_from_tiling % dedy_w == 0 and (mc_from_tiling * self.stride_h * self.stride_w
                        <= utils.NHW_MAX) and dedy_h % (mc_from_tiling // dedy_w) == 0:
                        n_is_hfactor_val = mc_from_tiling // dedy_w
                    else:
                        n_is_hfactor_val = (mc_from_tiling - utils.FP16_M) // dedy_w
                else:
                    n_is_hfactor_val = (mc_from_tiling - utils.FP16_M) // dedy_w
                n_is_hfactor = min(n_is_hfactor_val, max_n_is_hfactor)
                dilate_l0c_m = dx_w * n_is_hfactor * self.stride_h
                cub_dilate_size = dilate_l0c_m * nc_factor * n0 * cub_db_flag * cub_dtype_bit
                cub_dilate_size *= (1 + fused_double_operand_num)
                if (cub_size + cub_dilate_size + bias_size) > ub_size_limit:
                    return False
        return True

    def _check_tiling_al0(self, tiling_mess):
        """
        Check if tiling in repository al0 space is legal

        Parameters
        ----------
        tiling_mess: shape and tiling retrieved from repository
        Returns
        -------
        tiling_valid_flag: If true means it's legal
        """
        tiling_tmp = tiling_mess.get('tiling')
        l0_shape = "AL0_matrix"
        l0_space = tbe_platform_info.get_soc_spec("L0A_SIZE")
        row = tiling_tmp.get(l0_shape)[0]
        col = tiling_tmp.get(l0_shape)[1]
        group = tiling_tmp.get(l0_shape)[5]
        if row == 0 or col == 0:
            return False
        l0_dtype_bit = BIT_RATIO_DICT.get(self.a_type)
        data_amount_l0 = (
            row
            * col
            * tiling_tmp.get(l0_shape)[2]
            * tiling_tmp.get(l0_shape)[3]
            * group
            * l0_dtype_bit
        )
        if isinstance(data_amount_l0, int) and data_amount_l0 > l0_space:
            DynamicConv2dBpInputParams.dynamic_para["correct_range_flag"] = True
            return False
        return True

    def _modify_tiling_for_large_m(self, tiling_mess):
        """
        modify tiling for case with large m when stride > 1, only for kh == 1 and kw == 1

        Parameters
        ----------
        tiling_mess: tiling message with tiling and info

        Returns
        -------
        tiling_mess
        """
        tiling = tiling_mess.get("tiling")
        ori_m = tiling_mess.get("C_shape")[2] * tiling_mess.get("C_shape")[3]
        if ((self.stride_h > 1 or self.stride_w > 1) and
            self.k_h == 1 and self.k_w == 1 and ori_m > LARGE_M and tiling["AL0_matrix"][0] == 1):
            max_core_num = tbe_platform_info.get_soc_spec("CORE_NUM")
            l0a_size = tbe_platform_info.get_soc_spec("L0A_SIZE") // BIT_RATIO_DICT.get(self.a_type)
            l0c_size = tbe_platform_info.get_soc_spec("L0C_SIZE") // BIT_RATIO_DICT.get(self.c_type)
            tiling_db = tiling.get("manual_pingpong_buffer")

            # use max core nums
            core_num = reduce(lambda x, y: x * y, tiling["block_dim"])
            if core_num < max_core_num:
                tiling["block_dim"][2] = max_core_num // (core_num // tiling["block_dim"][2])

            m_per_core = ori_m // tiling["block_dim"][2]

            # use max buffer size
            modified_flag = False
            while (self.check_tiling_ub(tiling_mess) and m_per_core > tiling["CUB_matrix"][1] * 16
                and reduce(lambda x, y: x * y, tiling["AL0_matrix"]) * tiling_db.get("AL0_pbuffer") < l0a_size
                and reduce(lambda x, y: x * y, tiling["CL0_matrix"]) * tiling_db.get("CL0_pbuffer") < l0c_size
                and self.check_tiling_match(tiling, tiling_mess.get("C_shape")[3], tiling_mess.get("C_shape")[2])):
                tiling["CUB_matrix"][1] += 1
                tiling["AL0_matrix"][0] = tiling["CL0_matrix"][1] = tiling["CUB_matrix"][1]
                tiling_mess["tiling"] = tiling
                modified_flag = True
            if modified_flag:
                tiling["CUB_matrix"][1] -= 1
                tiling["AL0_matrix"][0] = tiling["CL0_matrix"][1] = tiling["CUB_matrix"][1]

        tiling_mess["tiling"] = tiling
        return tiling_mess

    def get_costmodel_tiling(self, shape):
        """
        get tiling using cost model

        Parameters
        ----------
        shape: specified shape to get tiling

        Returns
        -------
        tiling: tiling retrieved by cost model
        """

        if "batch_n" in self.var_map:
            self.a_info[0] = shape if isinstance(shape, int) else shape[0]
            self.c_info[0] = shape if isinstance(shape, int) else shape[0]
        if "dx_h" in self.var_map:
            self.c_info[2] = shape[1]
            self.a_info[2] = self.get_output_h(self.c_info[2])
        if "dx_w" in self.var_map:
            self.c_info[3] = shape[2]
            self.a_info[3] = self.get_output_w(self.c_info[3])
        self.tiling_info["tiling_type"] = "cost_model_tiling"
        for pad in ("padl", "padr", "padu", "padd"):
            self.tiling_info[pad] = 0
        if self.pad_mode == "FIX":
            _, _, dy_h, dy_w, _ = self.a_info
            new_hw = (dy_h * self.stride_h, dy_w * self.stride_w)
            new_pad_before = (
                (self.k_h - 1) * self.dilate_h - self.cur_pads[2],
                (self.k_w - 1) * self.dilate_w - self.cur_pads[0]
            )
            pad_up_before, pad_left_before = new_pad_before

            _, _, dx_h, dx_w, _ = self.c_info
            new_pad_after = tuple(
                i - o - pb + (k - 1) * d
                for i, o, pb, k, d in zip(
                    (dx_h, dx_w),
                    new_hw,
                    new_pad_before,
                    (self.k_h, self.k_w),
                    (self.dilate_h, self.dilate_w)
                )
            )
            pad_down_after, pad_right_after = new_pad_after

            pad_up_before = (pad_up_before + abs(pad_up_before)) // 2
            pad_left_before = (pad_left_before + abs(pad_left_before)) // 2
            pad_down_after = (pad_down_after + abs(pad_down_after)) // 2
            pad_right_after = (pad_right_after + abs(pad_right_after)) // 2

            self.tiling_info["padl"] = pad_left_before
            self.tiling_info["padr"] = pad_right_after
            self.tiling_info["padu"] = pad_up_before
            self.tiling_info["padd"] = pad_down_after
            if self.k_h == 1 and self.k_w == 1 and self.cur_pads == [0, 0, 0, 0]:
                self.tiling_info["general_flag"] = False
            else:
                self.tiling_info["general_flag"] = True

        cost_seeds = get_tiling(self.tiling_info)
        tiling_mess = self._check_and_set_default_tiling(cost_seeds[0])
        tiling_mess = self._modify_tiling_for_large_m(tiling_mess)

        return tiling_mess

    def get_tiling_range(self, tiling_in, c_shape):
        """
        get the covered area of a tiling

        Parameters
        ----------
        tiling_in : dict, result of tiling fetch

        c_shape : list, size of fmap_shape

        Returns
        -------
        list, range covered for tiling_in
        """
        def _modify_w_range():
            """
            modify w_range ensure that m_tiling - nw > 16
            """
            split_range_flag = False
            fmap_w_tiling = w_range_max

            if ("dx_w" in paras.get("var_map") and self.k_h == 1 and self.k_w == 1
                and (self.pad_mode == "VAR" or sum(self.cur_pads) == 0)):
                dy_w_tiling = tiling_in.get("CL0_matrix")[1] * tiling_in.get("CL0_matrix")[2]
                dy_w = self.get_output_w(fmap_w)
                if self.pad_mode == "VAR":
                    fmap_w_tiling = min((dy_w_tiling - MIN_STEP - 1) * self.stride_w, fmap_w_tiling)
                else:
                    fmap_w_tiling = min(
                        (dy_w_tiling - MIN_STEP - 1 - self.k_w) * self.stride_w + self.k_w, fmap_w_tiling)
                if dy_w % 16 == 0 and dy_w > dy_w_tiling - MIN_STEP and dy_w_tiling > (MIN_STEP + self.k_w):
                    split_range_flag = True
                else:
                    fmap_w_tiling = max(fmap_w_tiling, fmap_w)
            return split_range_flag, fmap_w_tiling

        def _modify_max_range():
            """
            modify h_max and w_max according to the limit of ub buffer,
            ensure that aub + cub < ub buffer
            aub = ma * ka * db_flag * bit_num
            cub = mc * nc * m0 * n0 * db_flag * bit_num
            """
            if tiling_in.get("AUB_shape"):
                cub_buffer = (reduce(lambda x, y: x * y, tiling_in.get("CUB_matrix"))
                              * tiling_in.get("manual_pingpong_buffer").get("CUB_pbuffer")
                              * BIT_RATIO_DICT.get(self.c_type))
                tiling_k_aub = tiling_in.get("AUB_shape")[0] // (self.b_info[2] * self.b_info[3])
                m_aub_max = ((tbe_platform_info.get_soc_spec("UB_SIZE") - cub_buffer)
                             // BIT_RATIO_DICT.get(self.a_type)
                             // tiling_in.get("manual_pingpong_buffer").get("AUB_pbuffer")
                             // tiling_k_aub / (1 + 1 / self.stride_w))

                if tiling_in.get("AUB_shape")[1] >= 1:
                    w_range = min(w_range_max, max(m_aub_max // tiling_in.get("AUB_shape")[1], c_shape[3]))
                    return w_range
            return w_range_max

        def _get_perf_range(h_range_max, w_range_max):
            # modify range for curv performance line
            bool_check_case = utils.icd(
                utils.icd(utils.icd(fmap_h * fmap_w, tiling["block_dim"][2]), utils.FP16_M),
                tiling["AL0_matrix"][0]) <= tiling["AL1_shape"][1]

            if bool_check_case:
                range_max = tiling["AL1_shape"][1] * tiling["AL0_matrix"][0] * \
                            utils.FP16_M * tiling["block_dim"][2]
                if h_range_max * w_range_max > range_max:
                    return range_max // fmap_w, fmap_w
            return h_range_max, w_range_max

        tiling = self._preprocess_tiling(tiling_in)
        _, _, fmap_h, fmap_w, _ = c_shape

        paras = {
            "var_map": self.var_map,
            "k_h": self.k_h,
            "k_w": self.k_w,
            "pad_mode": self.pad_mode,
            "pads": self.cur_pads
        }
        n_range_min, n_range_max = self.get_batch_range(c_shape[0], paras)
        tiling_range = [n_range_min, n_range_max]
        # check tiling covering itself situation
        if not self.check_tiling_match(tiling, fmap_w, fmap_h) or fmap_h > utils.NHW_MAX or fmap_w > utils.NHW_MAX:
            return tiling_range + [0, 0, 0, 0]
        h_range_min, h_range_max = self.get_h_range(fmap_h, tiling, paras)
        tiling_range += [h_range_min, h_range_max]
        w_range_min, w_range_max = self.get_w_range(fmap_h, fmap_w, tiling, paras)
        split_range_flag, w_range_max = _modify_w_range()
        tiling_range += [w_range_min, w_range_max]
        if split_range_flag:
            tiling_range_self = tiling_range[:4] + [fmap_w, fmap_w]
            tiling_range_list = [tiling_range, tiling_range_self]

        if not tiling.get("AL1_shape"):
            if split_range_flag:
                return tiling_range_list
            return tiling_range

        tiling_range[-1] = _modify_max_range()
        tiling_range[3], tiling_range[5] = _get_perf_range(tiling_range[3], tiling_range[5])

        if split_range_flag:
            return tiling_range_list
        return tiling_range

    def assembly_case(self, tiling, coverage, cnt):
        """
        Configure dict of tiling strategy and coverage

        Parameters
        ----------
        tiling: dict, tiling from repository or cost model

        coverage: list of tuple, coverage of tiling

        cnt: serial number of tiling

        Returns
        -------
        dict: describe a tiling strategy
        """

        var_range = OrderedDict()
        if "batch_n" in self.var_map:
            var_range['batch_n'] = (utils.trans_to_int(coverage[0]), utils.trans_to_int(coverage[1]))
        if "dedy_h" in self.var_map:
            dx_h_low, dx_h_high = utils.trans_to_int(coverage[2]), utils.trans_to_int(coverage[3])
            dedy_h_low = self.get_output_h(dx_h_low)
            dedy_h_high = self.get_output_h(dx_h_high)
            var_range['dx_h'] = (dx_h_low, dx_h_high)
            var_range['dedy_h'] = (dedy_h_low, dedy_h_high)
        if "dedy_w" in self.var_map:
            dx_w_low, dx_w_high = utils.trans_to_int(coverage[4]), utils.trans_to_int(coverage[5])
            dedy_w_low = self.get_output_w(dx_w_low)
            dedy_w_high = self.get_output_w(dx_w_high)
            var_range['dx_w'] = (dx_w_low, dx_w_high)
            var_range['dedy_w'] = (dedy_w_low, dedy_w_high)
        correct_range_flag = DynamicConv2dBpInputParams.dynamic_para.get("correct_range_flag", False)

        return {"key": cnt, "tiling_strategy": tiling, "var_range": var_range, "correct_range_flag": correct_range_flag}

    def get_default_tiling(self, w_lower_bound=1):
        """
        get default tiling for unlimited range or special case

        Parameters
        ----------
        w_lower_bound: the min value of w when dynamic w

        Returns
        -------
        dict: default tiling for conv2d_bp_input
        """

        tiling = {}
        _, _, k_w, k_h, _ = self.b_info
        bit_dir = {
            "float32": 16,
            "int32": 16,
            "float16": 16,
            "int8": 32,
        }
        atype = self.tiling_info["A_dtype"]
        btype = self.tiling_info["B_dtype"]
        if atype in bit_dir.keys():
            k_al1 = k_w * k_h * 16
            k_al0 = bit_dir[atype]
        else:
            # default value 32
            k_al1 = 32
            k_al0 = 32

        if btype in bit_dir.keys():
            k_bl1 = bit_dir[atype]
            k_bl0 = bit_dir[atype]
        else:
            # default value 32
            k_bl1 = 32
            k_bl0 = 32

        k_aub = m_al0 = m_cl0 = 1
        if self.a_info[3] == -1:
            w_value = w_lower_bound
        else:
            w_value = self.a_info[3]
        if self.tiling_info["strideH_expand"] > 1 or self.tiling_info["strideW_expand"] > 1:
            if self.k_h == 1 and self.k_w == 1 and (self.pad_mode == "VAR" or sum(self.cur_pads) == 0):
                # when mmad, the min unit of M is a fmp's w
                if w_value % 16 == 0:
                    m_al0 = m_cl0 = w_value // utils.FP16_M
                else:
                    # add one is needed by buffer_tile of ub
                    m_al0 = m_cl0 = utils.icd(w_value, utils.FP16_M) + 1
            else:
                k_aub = k_w * k_h * 16
        n_min = 1
        group_cl0 = 1

        tiling["AUB_shape"] = [k_aub, 1, 1, 1] if k_aub != 1 else None
        tiling["BUB_shape"] = None
        tiling["AL1_shape"] = [k_al1, 1, 1, 1]
        tiling["BL1_shape"] = [k_bl1, 1, 1, 1]
        tiling["AL0_matrix"] = [m_al0, 1, 16, k_al0, 1, 1]
        tiling["BL0_matrix"] = [1, n_min, 16, k_bl0, 1, 1]
        tiling["CL0_matrix"] = [n_min, m_cl0, 16, 16, 1, group_cl0]
        tiling["CUB_matrix"] = [n_min, m_cl0, 16, 16, 1, group_cl0]
        tiling["block_dim"] = [1, 1, 1, 1]
        tiling["n_bef_batch_flag"] = 0
        tiling["n_bef_group_flag"] = 0
        tiling["batch_bef_group_fla"] = 0
        tiling["A_overhead_opt_flag"] = 0
        tiling["B_overhead_opt_flag"] = 0
        tiling["AUB_channel_wise_flag"] = None
        tiling["BUB_channel_wise_flag"] = None
        tiling["CUB_channel_wise_flag"] = None
        tiling["manual_pingpong_buffer"] = {
            'AUB_pbuffer': 1,
            'BUB_pbuffer': 1,
            'AL1_pbuffer': 1,
            'BL1_pbuffer': 1,
            'AL0_pbuffer': 1,
            'BL0_pbuffer': 1,
            'CL0_pbuffer': 1,
            'CUB_pbuffer': 1,
            'UBG_pbuffer': 1,
        }
        return tiling

    def _check_and_set_default_tiling(self, tiling_in):
        if tiling_in.get("tiling").get("AL0_matrix")[2] == 32:
            tiling_in = {"tiling": self.get_default_tiling(), "A_shape": self.a_info,
                        "B_shape": self.b_info, "C_shape": self.c_info}
        while not self.check_tiling_ub(tiling_in):
            if tiling_in.get("tiling").get("manual_pingpong_buffer").get("CUB_pbuffer") == 2:
                tiling_in["tiling"]["manual_pingpong_buffer"]["CUB_pbuffer"] = 1
                continue
            _, mc_factor, m0, _ = tiling_in.get("tiling").get("CUB_matrix")[:4]
            if self.k_h == 1 and self.k_w == 1:
                if (mc_factor * m0 - utils.FP16_M) > self.a_info[3]:
                    tiling_in["tiling"]["CUB_matrix"][1] -= 1
                    tiling_in["tiling"]["CL0_matrix"][1] -= 1
                    tiling_in["tiling"]["AL0_matrix"][0] -= 1
                else:
                    return {"tiling": self.get_default_tiling(), "A_shape": self.a_info,
                            "B_shape": self.b_info, "C_shape": self.c_info}
            elif mc_factor > 1:
                tiling_in["tiling"]["CUB_matrix"][1] -= 1
                tiling_in["tiling"]["CL0_matrix"][1] -= 1
                tiling_in["tiling"]["AL0_matrix"][0] -= 1
            else:
                return {"tiling": self.get_default_tiling(), "A_shape": self.a_info,
                        "B_shape": self.b_info, "C_shape": self.c_info}
        return tiling_in

    def _get_calc_info(self):
        self._convert_type(self.a_info, self.b_info, self.c_info)
        self.k_h, self.k_w = self.b_info[2:4]
        self.k_cout = self.b_info[1] * self.b_info[4]
        self.k_cin = self.b_info[0]
        self.stride_h, self.stride_w = self.tiling_info["strideH_expand"], \
                                       self.tiling_info["strideW_expand"]
        self.dilate_h, self.dilate_w = self.tiling_info["dilationH"], \
                                       self.tiling_info["dilationW"],

        if isinstance(self.tiling_info["padl"], Expr) or isinstance(self.tiling_info["padu"], Expr):
            self.pad_mode = "VAR"
        else:
            self.pad_mode = "FIX"
        self.cur_pads = []
        for pad in ("padl", "padr", "padu", "padd"):
            self.cur_pads.append(self.tiling_info[pad])
            self.tiling_info[pad] = -1

    def _preprocess_tiling(self, tiling_in):
        """
        preprocess tiling for get tiling range
        """
        tiling = copy.deepcopy(tiling_in)
        if tiling["AL1_shape"]:
            tiling["AL1_shape"][0] = tiling["AL1_shape"][0] // \
                (self.k_h * self.k_w * utils.CUBE_SIZE)
        if tiling["BL1_shape"]:
            tiling["BL1_shape"][0] = tiling["BL1_shape"][0] // \
                (self.k_h * self.k_w * utils.CUBE_SIZE)
        return tiling

    def _get_al1_bound(self, tiling, curent_size):
        """
        get al1 bound info

        Parameters
        ----------
        tiling : dict, result of tiling fetch

        current_size : int, size of h,w

        Returns
        -------
        int, al1_load_length (al1_bound)

        """

        # shape info
        out_w = curent_size
        w_i = self.get_output_w(out_w) * self.stride_w

        if len(tiling['AL1_shape']) == 1:
            tiling['AL1_shape'].append(1)

        # M axis theorically loading length in al1
        al1_m_data = tiling['CL0_matrix'][1] * utils.FP16_M * tiling['AL1_shape'][1]

        # load2d instructions refer to data_mov with raw lens
        if (self.pad_mode == "VAR" or sum(self.cur_pads) == 0) and \
                self.k_h * self.k_w == 1:
            return al1_m_data

        # tiling load lens less than out_w, nned to load a full line
        if al1_m_data < out_w:
            l1_ho = 1 if out_w % al1_m_data == 0 else 2
        else:
            # load3d instructions refer to load extra lines with pad/stride/filter
            if al1_m_data % out_w == 0:
                # full line could load without extra lines
                extend_h = 0
            elif (al1_m_data * 2) % out_w == 0:
                # every 2 load3d covered only 1 extra line
                extend_h = 1
            else:
                # other situations need 2 extra lines in case
                extend_h = 2
            l1_ho = al1_m_data // out_w + extend_h

        # calculate input lines (hi) from output lines (ho)
        li_hi = self.k_h + (l1_ho - 1)

        return li_hi * w_i

    def check_tiling_match(self, tiling, current_w, current_h):
        """
        check whether this tiling matches the shape

        Parameters
        ----------
        tiling : dict, result of tiling fetch

        current_size : int, size of h,w

        Returns
        -------
        bool, True: match
              False: do not match

        """

        if not tiling.get("AL1_shape"):
            return True

        # shape info
        fmap_w = current_w

        # get M axis length in al1
        al1_bound = self._get_al1_bound(tiling, fmap_w)

        # fmap size in L1 ( M * K * db * 2byte)
        fmap_l1_size = utils.FP16_SIZE * al1_bound * tiling['AL1_shape'][0] * \
            utils.FP16_K * tiling['manual_pingpong_buffer']['AL1_pbuffer']

        # filter size
        if tiling['BL1_shape'] is None:
            # not using BL1
            filter_l1_size = 0
        elif len(tiling['BL1_shape']) == 0:
            # fully load in BL1
            filter_l1_size = utils.FP16_SIZE * self.k_cout * self.k_cin * self.k_h * \
                self.k_w // tiling['block_dim'][1]
        else:
            # fmap size in L1 ( K * N * db * 2byte)
            filter_l1_size = utils.FP16_SIZE * tiling['BL1_shape'][1] * \
                tiling['CL0_matrix'][0] * utils.FP16_N * tiling['BL1_shape'][0] * \
                utils.FP16_K * self.k_h * self.k_w * \
                tiling['manual_pingpong_buffer']['BL1_pbuffer']

        l1_buffer_flag = True if fmap_l1_size + filter_l1_size <= utils.L1BUFFER else False
        
        A_shape = self.a_info[:3] + [self.get_output_w(fmap_w)] + [self.a_info[-1]]
        C_shape = self.c_info[:3] + [fmap_w] + [self.c_info[-1]]
        tiling_mess = {"tiling": tiling, "A_shape": A_shape, "B_shape": self.b_info, "C_shape": C_shape,
                       "fused_double_operand_num": self.tiling_info.get("fused_double_operand_num"),
                       "bias_flag": self.tiling_info.get("bias_flag")}
        ub_buffer_flag = self.check_tiling_ub(tiling_mess)
        
        return l1_buffer_flag and ub_buffer_flag

    def get_output_h(self, fmap_h, stride=None):
        """
        calculate output h
        """
        if not fmap_h:
            return fmap_h
        if not stride:
            stride = self.stride_h
        if self.pad_mode == "VAR":
            return utils.icd(fmap_h, stride)
        return (fmap_h + self.cur_pads[2] + self.cur_pads[3] - self.dilate_h * (self.k_h - 1) - 1) // stride + 1

    def get_output_w(self, fmap_w, stride=None):
        """
        calculate output w
        """
        if not fmap_w:
            return fmap_w
        if not stride:
            stride = self.stride_w
        if self.pad_mode == "VAR":
            return utils.icd(fmap_w, stride)
        return (fmap_w + self.cur_pads[0] + self.cur_pads[1] - self.dilate_w * (self.k_w - 1) - 1) // stride + 1