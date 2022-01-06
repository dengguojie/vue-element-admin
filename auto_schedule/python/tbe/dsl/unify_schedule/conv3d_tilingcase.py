#!/usr/bin/env python
# -*- coding:utf-8 -*-
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
conv3d tiling case
"""

import copy
import math
import json
from collections import OrderedDict
from functools import reduce

from tbe.common.platform import platform_info as tbe_platform_info
from tbe.common.tiling.get_tiling import get_tiling
from tbe.common.context import get_context
from tbe.dsl.base.operation import add_compile_info
from tbe.dsl.base.operation import get_compile_info
from tbe.dsl.base.operation import register_tiling_case
from tbe.dsl.base.operation import register_build_pointcut
from tbe.dsl.base.operation import get_context as op_get_context
from tbe.dsl.base.operation import get_te_var
from tbe.dsl.compute import util as te_util
from tbe.dsl.compute.conv3d_compute import Conv3DParam
from tbe import tvm
from tbe.tvm.expr import Expr

from .cube_tilingcase import CubeTilingOp
from .cube_tilingcase import TilingSelection
from .cube_tilingcase import TilingUtils as utils
from .constants import Pattern


H_RANGE = 4096
W_RANGE = 4096
W_DELTA = 1
D_DELTA = 1
H_LEN = 400
W_LEN = 400
D_LEN = 400
VALID_TILING_NUM = 32
N_BASE = 2
BIT_RATIO_DICT = {"float32": 4, "float16": 2}


def _parse_fuzz_build_range(info_list):
    """
    parse multiple range segment from json string

    Notice
    ----------
    for conv3d only parse input range

    Parameters
    ----------
    info_list: list support info
        [{
            "inputs": [{
                "index": 0,
                "tensor": [{
                    "range": [
                        [1, 1],
                        [8, 15],
                        [32, 63],
                        [32, 63],
                        [64, 64]
                    ],
                    "shape": [-1, -1, -1, -1, 64]
                }]
            }]
        }]

    Returns
    -------
    range_list: list of 5d range
    """
    target_index = 0
    range_list = []
    for item in info_list:
        inputs = item.get("inputs")
        for input_tensor in inputs:
            invalid = (not isinstance(input_tensor, dict)) or input_tensor.get("index") != target_index
            if invalid:
                continue
            input_range = input_tensor.get("tensor")[0].get("range")
            if input_range:
                for axis_range in input_range:
                    invalid = (not isinstance(axis_range, list)) \
                              or len(axis_range) != 2 \
                              or axis_range[0] > axis_range[1] \
                              or axis_range[0] < 1
                    if invalid:
                        raise RuntimeError("invalid range {}".format(str(axis_range)))
                range_list.append(input_range)
    return range_list


def gen_support_info(range_x):
    """
    kernel list support info part

    Notice
    ------
    only need to set inputs with range

    Parameters
    ----------
    range_x: list
    -   input x range

    Returns
    -------
    support_info: dict
    """
    support_info = {}
    inputs = []
    item = {}
    item["index"] = 0
    item["tensor"] = []
    tensor_info = {}
    ori_tensors = Conv3DParam.para_dict.get("ori_tensors")
    ori_tensors_input = ori_tensors.get("fmap")
    ori_shape = ori_tensors_input.get("ori_shape")
    tensor_info["shape"] = ori_shape
    x_format = ori_tensors_input.get("ori_format")
    range_valid = [[0, 0]] * 5
    # range_x : NDHW, @see get_tiling_range
    range_valid[x_format.find("N")] = list(range_x[0])
    range_valid[x_format.find("D")] = list(range_x[1])
    range_valid[x_format.find("H")] = list(range_x[2])
    range_valid[x_format.find("W")] = list(range_x[3])
    range_valid[x_format.find("C")] = [ori_shape[x_format.find("C")]] * 2
    tensor_info["range"] = range_valid
    item.get("tensor").append(tensor_info)
    inputs.append(item)
    support_info["inputs"] = inputs
    return support_info


def add_covered_shape_range(compile_info):
    """
    tiling_case func for dynamic shape conv3d

    Parameters
    ----------
    compile_info: dict
    -   tiling range info

    Returns
    -------
    info_list: dict
    -   support info and compile info pair
    max_kernel_id: int
    -   last kernel id
    """
    id_list = list(compile_info["block_dim"].keys())
    id_list.sort()
    te_vars = []
    info_list = []
    for cpt in op_get_context().get_computes():
        te_vars += cpt.get_vars()
    var_list = [var.get_name() for var in te_vars]
    for kernel_id, _ in compile_info["block_dim"].items():
        new_compile = compile_info.copy()
        for keys, value in new_compile.items():
            if isinstance(value, dict):
                value = {} if value.get(kernel_id) is None else {kernel_id: value[kernel_id]}
                new_compile[keys] = value
        new_compile["kernelId"] = kernel_id
        new_compile["_vars"] = {kernel_id: var_list}
        range_x = new_compile["repo_range"].get(kernel_id) or new_compile["cost_range"].get(kernel_id)
        new_range = [range_x[:2], range_x[2:4], range_x[4:6], range_x[6:8]]
        new_support = gen_support_info(new_range)
        info_list.append({"supportInfo": new_support, "compileInfo": new_compile})
    return info_list, id_list[-1]


@register_build_pointcut(pattern=Pattern.CONV3D)
def build_pointcut_conv3d(func, *args, **kwargs):
    """
    kernel info process before build

    Notice
    ------
    kernel_info: dict with support info and compile info
    -   {
    -       "supportInfo": {
    -           "inputs": [{
    -               "index": 0,
    -               "tensor": [{
    -               "range": [
    -                   [1, 1],
    -                   [8, 15],
    -                   [32, 63],
    -                   [32, 63],
    -                   [64, 64]
    -               ],
    -               "shape": [-1, -1, -1, -1, 64]
    -               }]
    -           }]
    -       },
    -       "compileInfo": {
    -           "_pattern": "conv3d",
    -           "tiling_type": "dynamic_tiling",
    -           "repo_seeds": {},
    -           "repo_range": {},
    -           "cost_range": {},
    -           "block_dim": {
    -               1: 32
    -           },
    -           "_vars": {
    -               0: ["batch_n", "fmap_d", "d_out", "fmap_h", "h_out", "fmap_w", "w_out"]
    -           }
    -       }
    -   }

    Parameters
    ----------
    func: funtions
    -   build process
    args: list
    -   function input args
    kwargs: dict
    -   function input args and value

    Returns
    -------
    None
    """
    fuzz_build = (get_context().get_build_type() == "fuzzily_build")
    if fuzz_build:  # set kernel info
        info_list, max_kernel_id = add_covered_shape_range(get_compile_info())
        get_context().add_build_json_result("kernelList", info_list)
        get_context().add_build_json_result("maxKernelId", max_kernel_id)
    func(*args, **kwargs)


def query_tiling_cases(tgt_list, conv_info, max_kernel_id, var_names):
    """
    do query tiling cases for dynamic shape conv3d

    Parameters
    ----------
    tgt_list : list of dict, each dict for a dynamic dim
    conv_info: tiling_info_dict
    max_kernel_id: the max value of kernel id
    var_names: keys of dynamic dims

    Returns
    -------
    list of dict, each dict for a tiling case
    """
    all_compile_info = {}
    tiling_cases = []
    for tgt in tgt_list:
        tiling_op = Conv3dTiling(conv_info, Conv3DParam.var_map)
        selector = TilingSelection(tiling_op, max_kernel_id)
        tiling_cases += selector.calc_tiling(tgt, var_names)
        fuzz_build = (get_context().get_build_type() == "fuzzily_build")
        if fuzz_build:
            tgt_ndhw = []
            for var_name in var_names:
                tgt_ndhw.extend(tgt[var_name])
            current_info = get_compile_info().copy()
            id_list = list(current_info["block_dim"].keys())
            id_list.sort()
            max_kernel_id = id_list[-1] + 1
            for range_key in ["repo_range", "cost_range"]:
                if isinstance(current_info.get(range_key), dict):
                    for kernel_id, range_x in current_info[range_key].items():
                        new_range = []
                        for index, dim_value in enumerate(range_x):
                            if index not in (0, 2, 4, 6):
                                new_range.append(tgt_ndhw[index] if dim_value > tgt_ndhw[index] else dim_value)
                            else:
                                new_range.append(tgt_ndhw[index] if dim_value < tgt_ndhw[index] else dim_value)
                        current_info[range_key][kernel_id] = new_range
            if all_compile_info:
                for key, value in current_info.items():
                    if isinstance(all_compile_info.get(key), dict) and isinstance(value, dict):
                        new_item = all_compile_info.get(key)
                        new_item.update(value)
                        all_compile_info[key] = new_item
                        add_compile_info(key, all_compile_info.get(key))
            else:
                all_compile_info = current_info
        group_dict = Conv3DParam.tensor_map.get("group_dict")
        fmap_cin1_g = utils.icd(group_dict["cin_ori"], conv_info.get("a_shape")[-1])
        add_compile_info("fmap_c1", fmap_cin1_g)
    return tiling_cases


def get_tiling_cases(tgt_list, conv_info, var_names):
    """
    get tiling cases for dynamic shape conv3d

    Parameters
    ----------
    tgt_list : list of dict, each dict for a dynamic dim
    conv_info: tiling_info_dict
    var_names: keys of dynamic dims

    Returns
    -------
    list of dict, each dict for a tiling case
    """
    para_dict = Conv3DParam.para_dict
    max_kernel_id = None
    fuzz_build = (get_context().get_build_type() == "fuzzily_build")
    if fuzz_build:
        missing_support_info = get_context().get_addition("missing_support_info")
        if len(missing_support_info) > 0:
            missing_support_list = _parse_fuzz_build_range(json.loads(missing_support_info))
            if len(missing_support_list) > 0:
                tgt_list.clear()  # clear the old data and only deal with fuzz shape range
                fmap_ori_format = para_dict.get("ori_tensors").get("fmap").get("ori_format")
                pos_list = [fmap_ori_format.find("N"),
                            fmap_ori_format.find("D"),
                            fmap_ori_format.find("H"),
                            fmap_ori_format.find("W")]
                for item in missing_support_list:
                    if item:
                        fuzz_area = {}
                        for var, index in zip(var_names, pos_list):
                            fuzz_area[var] = tuple(item[index])
                        tgt_list.append(fuzz_area)
        kernel_id = get_context().get_addition("max_kernel_id")
        valid = isinstance(kernel_id, int) and kernel_id > -2
        if valid:
            max_kernel_id = kernel_id + 1

    tiling_cases = query_tiling_cases(tgt_list, conv_info, max_kernel_id, var_names)
    return tiling_cases


# noinspection PyUnusedLocal
@register_tiling_case(pattern=Pattern.CONV3D)
def calc_conv3d(outs, option=None):
    """
    tiling_case func for dynamic shape conv3d

    Parameters
    ----------
    outs: tvm tensor or list of tvm tensor, results for tvm compute

    Returns
    -------
    list of dict, each dict for a tiling case
    """
    var_names = ('batch_n', 'fmap_d', 'fmap_h', 'fmap_w')
    tgt_list = []
    tgt_area = {}
    conv_info = Conv3DParam.tiling_info_dict
    conv_info["dynamic_shape_flag"] = True
    shape_dict = {"batch_n": conv_info.get("a_shape")[0],
                  "fmap_d": conv_info.get("a_shape")[1],
                  "fmap_h": conv_info.get("a_shape")[3],
                  "fmap_w": conv_info.get("a_shape")[4]}
    for var_name in var_names:
        if get_te_var(var_name):
            tgt_area[var_name] = tuple(get_te_var(var_name).get_bound())
        else:
            tgt_area[var_name] = (int(shape_dict.get(var_name)), int(shape_dict.get(var_name)))

    tgt_list.append(tgt_area) # deal with dynamic shape range by default

    tiling_cases = get_tiling_cases(tgt_list, conv_info, var_names)
    return tiling_cases


class Conv3dTiling(CubeTilingOp):
    """
    conv3d tiling class
    """
    def __init__(self, tiling_info, var_map):
        super().__init__(tiling_info, var_map)
        self.a_info = tiling_info['a_shape']
        self.b_info = tiling_info['b_shape']
        self.c_type = tiling_info["c_dtype"]
        self._get_calc_info()
        self.key = 'A_shape'
        self.op_type = "convolution_3d"
        self.var_map = var_map

    def get_repo_tiling(self):
        tiling_list = get_tiling(self.tiling_info)
        res_list = []
        for tiling in tiling_list:
            t_padh, t_padt, t_padu, t_padd, t_padl, t_padr = tiling.get('pad')
            t_out_d = (tiling.get('A_shape')[1] + t_padh + t_padt - self.dilation_d * (self.k_d - 1) - 1) \
                      // self.stride_d + 1
            t_out_h = (tiling.get('A_shape')[3] + t_padu + t_padd - self.dilation_h * (self.k_h - 1) - 1) \
                      // self.stride_h + 1
            t_out_w = (tiling.get('A_shape')[4] + t_padl + t_padr - self.dilation_w * (self.k_w - 1) - 1) \
                      // self.stride_w + 1
            out_d = self._get_output_d(tiling.get('A_shape')[1])
            out_h = self._get_output_h(tiling.get('A_shape')[3])
            out_w = self._get_output_w(tiling.get('A_shape')[4])
            if out_d == t_out_d and out_h == t_out_h and out_w == t_out_w:
                res_list.append(tiling)
        return res_list

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

        if 'batch_n' in self.var_map:
            self.a_info[0] = shape if isinstance(shape, int) else shape[0]
        if 'fmap_d' in self.var_map:
            self.a_info[1] = shape[1]
        if 'fmap_h' in self.var_map:
            self.a_info[3] = shape[2]
        if 'fmap_w' in self.var_map:
            self.a_info[4] = shape[3]
        if self.pad_mode == "VAR":
            self.tiling_info["pad"] = \
                self._calc_pads(shape[1], shape[2], shape[3])
        self.tiling_info["tiling_type"] = "cost_model_tiling"

        tiling = get_tiling(self.tiling_info)[0]
        return tiling

    def get_default_tiling(self):
        """
        get default tiling

        Parameters
        ----------
        None

        Returns
        -------
        default tiling
        """

        def _handle_block_dim():
            """
            avoid cyclomatic complexity, handle block_dim
            """
            tiling["block_dim"] = [1, 1, 1, 1]
            device_core_num = tbe_platform_info.get_soc_spec("CORE_NUM")
            if (self.a_info[0] > 1) and (device_core_num > 1):
                if self.a_info[0] <= device_core_num:
                    tiling.get("block_dim")[0] = self.a_info[0]
                else:
                    for i in range(device_core_num, 0, -1):
                        if self.a_info[0] % i == 0:
                            break
                    tiling.get("block_dim")[0] = i
            else:
                tiling.get("block_dim")[0] = 1

        _, _, _, filter_h, filter_w, _ = self.b_info
        tiling = {}
        tiling_m = 1
        tiling_k = 1
        tiling_n = 1
        tiling["AL1_shape"] = [filter_h * filter_w * utils.FP16_K, 1, 1, 1]
        tiling["BL1_shape"] = None
        tiling["AL0_matrix"] = [tiling_m, tiling_k, utils.FP16_M, utils.FP16_K, 1, 1]
        tiling["BL0_matrix"] = [tiling_k, tiling_n, utils.FP16_N, utils.FP16_K, 1, 1]
        tiling["CL0_matrix"] = [tiling_n, tiling_m, utils.FP16_M, utils.FP16_N, 1, 1]
        tiling["CUB_matrix"] = [tiling_n, tiling_m, utils.FP16_M, utils.FP16_N, 1, 1]
        tiling["AUB_shape"] = tiling.get("AL1_shape")
        tiling["BUB_shape"] = None
        tiling["manual_pingpong_buffer"] = {'AL1_pbuffer': 1,
                                            'BL1_pbuffer': 1,
                                            'AL0_pbuffer': 1,
                                            'BL0_pbuffer': 1,
                                            'CL0_pbuffer': 1,
                                            'CUB_pbuffer': 1,
                                            'UBG_pbuffer': 1,
                                            'AUB_pbuffer': 1}
        tiling["A_overhead_opt_flag"] = False
        tiling["B_overhead_opt_flag"] = False
        tiling["CUB_channel_wise_flag"] = True
        tiling["n_bef_batch_flag"] = False
        _handle_block_dim()

        return tiling

    def schedule_handle(self, block_dims, tiling):
        """
        schedule handle

        Parameters
        ----------
        block_dims: block dim
        tiling: tiling info

        Returns
        -------
        the min of block_dims and al1_factor
        """
        # calculate the actual block_dim_m in the dynamic batch
        out_img_shape = self._get_output_h(self.a_info[3])*self._get_output_w(self.a_info[4])
        c_tiling_factor = tiling['CL0_matrix'][1]*tiling['CL0_matrix'][2]
        c_factor = te_util.int_ceil_div(out_img_shape, c_tiling_factor)
        if len(tiling['AL1_shape']) == 1:
            tiling['AL1_shape'] = tiling['AL1_shape'] + [1]
        if tiling['AL1_shape']:
            al1_factor = te_util.int_ceil_div(c_factor, tiling['AL1_shape'][1])
        else:
            al1_factor = 1

        return tvm.min(block_dims, al1_factor)

    def assembly_case(self, tiling_strategy, covered, cnt):
        var_range = OrderedDict()
        if 'batch_n' in self.var_map:
            var_range['batch_n'] = (covered[0], covered[1])

        if 'fmap_d' in self.var_map:
            var_range['fmap_d'] = (covered[2], covered[3])
            var_range['d_out'] = (self._get_output_d(var_range['fmap_d'][0]),
                               self._get_output_d(var_range['fmap_d'][1]))

        if 'fmap_h' in self.var_map:
            var_range['fmap_h'] = (covered[4], covered[5])
            var_range['h_out'] = (self._get_output_h(var_range['fmap_h'][0]),
                               self._get_output_h(var_range['fmap_h'][1]))

        if 'fmap_w' in self.var_map:
            var_range['fmap_w'] = (covered[6], covered[7])
            var_range['w_out'] = (self._get_output_w(var_range['fmap_w'][0]),
                               self._get_output_w(var_range['fmap_w'][1]))

        block_dim_multi = tiling_strategy["BUB_shape"][0] if tiling_strategy["BUB_shape"] else 1
        block_dims = block_dim_multi *\
            reduce(lambda x, y: x * y, tiling_strategy["block_dim"])
        if 'fmap_h' not in self.var_map and 'fmap_w' not in self.var_map:
            new_block_dims = copy.deepcopy(tiling_strategy['block_dim'])
            new_tiling = copy.deepcopy(tiling_strategy)
            new_block_dims[2] = int(self.schedule_handle(new_block_dims[2], new_tiling))
            block_dims = block_dim_multi *\
                reduce(lambda x, y: x * y, new_block_dims)
        if tiling_strategy["AL0_matrix"][2] == VALID_TILING_NUM:
            batch_size = self.a_info[0]
            device_core_num = tbe_platform_info.get_soc_spec("CORE_NUM")
            if 'batch_n' in self.var_map:
                block_dims = device_core_num
            elif te_util.get_and_res(batch_size > 1, device_core_num > 1):
                if batch_size <= device_core_num:
                    block_dims = batch_size
                else:
                    for i in range(device_core_num, 0, -1):
                        if batch_size % i == 0:
                            break
                    block_dims = i
            else:
                block_dims = 1

        correct_range_flag = Conv3DParam.dynamic_para.get("correct_range_flag")
        return {"key": cnt, "tiling_strategy": tiling_strategy, "var_range": var_range,
                "block_dim": block_dims, "correct_range_flag": correct_range_flag}

    def _get_calc_info(self):
        self._convert_type(self.a_info, self.b_info)
        self.k_d, self.k_h, self.k_w = self.b_info[1], self.b_info[3], self.b_info[4]
        self.k_cin = self.b_info[2] * self.b_info[5]
        self.k_cout = self.b_info[0]

        self.pad_mode = "FIX"
        # currently, in dynamic_dhw, when padding is SAME, pad_mode is "VAR"
        if any(isinstance(i, Expr) for i in self.tiling_info["pad"]):
            self.pad_mode = "VAR"
            self.tiling_info["pad"] = [-1, -1, -1, -1, -1, -1]
        self.padf, self.padb, self.padu, self.padd, self.padl, self.padr =\
            self.tiling_info["pad"]
        self.stride_d, self.stride_h, self.stride_w = self.tiling_info["stride"]
        self.dilation_d, self.dilation_h, self.dilation_w = \
            self.tiling_info["dilation"]
        self.k_d_dilation = (self.k_d - 1) * self.dilation_d + 1
        self.k_h_dilation = (self.k_h - 1) * self.dilation_h + 1
        self.k_w_dilation = (self.k_w - 1) * self.dilation_w + 1

    def _get_output_d(self, d_in):
        if not d_in:
            return None
        if self.pad_mode == "VAR":
            return utils.icd(d_in, self.stride_d)
        return (d_in + self.padf + self.padb - self.dilation_d *
                (self.k_d - 1) - 1) // self.stride_d + 1

    def _get_output_h(self, h_in):
        if not h_in:
            return None
        if self.pad_mode == "VAR":
            return utils.icd(h_in, self.stride_h)
        return (h_in + self.padu + self.padd - self.dilation_h *
                (self.k_h - 1) - 1) // self.stride_h + 1

    def _get_output_w(self, w_in):
        if not w_in:
            return None
        if self.pad_mode == "VAR":
            return utils.icd(w_in, self.stride_w)
        return (w_in + self.padl + self.padr - self.dilation_w *
                (self.k_w - 1) - 1) // self.stride_w + 1

    def _calc_pads(self, d_in, h_in, w_in):
        pad_d = utils.align(d_in, self.stride_d) - self.stride_d + \
            self.k_d_dilation - d_in
        pad_d = max(pad_d, 0)
        pad_front = pad_d // 2
        pad_back = pad_d - pad_front

        pad_h = utils.align(h_in, self.stride_h) - self.stride_h + \
            self.k_h_dilation - h_in
        pad_h = max(pad_h, 0)
        pad_up = pad_h // 2
        pad_down = pad_h - pad_up

        pad_w = utils.align(w_in, self.stride_w) - self.stride_w + \
            self.k_w_dilation - w_in
        pad_w = max(pad_w, 0)
        pad_left = pad_w // 2
        pad_right = pad_w - pad_left
        return [pad_front, pad_back, pad_up, pad_down, pad_left, pad_right]

    def _preprocess_tiling(self, tiling_in):
        """
        preprocess tiling for get tiling range
        """

        tiling = copy.deepcopy(tiling_in)
        if tiling["AL1_shape"]:
            tiling["AL1_shape"][0] = tiling["AL1_shape"][0] * tiling["AL1_shape"][-1] // \
                (self.k_h * self.k_w * utils.CUBE_SIZE)
        if tiling["BL1_shape"]:
            tiling["BL1_shape"][0] = tiling["BL1_shape"][0] * tiling["BL1_shape"][-1] // \
                (self.k_h * self.k_w * utils.CUBE_SIZE)
        return tiling

    def _get_al1_bound(self, tiling, current_size):
        """
        get al1 bound info

        Parameters
        ----------
        tiling : dict, result of tiling fetch

        current_size : int, size of w

        Returns
        -------
        int, al1_load_length (al1_bound) in M axis

        """

        # shape info
        w_i = current_size
        out_w = self._get_output_w(w_i)

        pad_lis = [self.padf, self.padb, self.padu, self.padd, self.padl, self.padr]
        strideh_opti_flag = self.k_h == 1 and self.stride_h > 1

        if len(tiling['AL1_shape']) == 1:
            tiling['AL1_shape'].append(1)

        # M axis theorically loading length in al1
        al1_m_data = tiling['CL0_matrix'][1] * utils.FP16_M * tiling['AL1_shape'][1]

        # load2d instructions refer to data_mov with raw lens
        if (self.pad_mode == "VAR" or sum(pad_lis) == 0) \
            and (self.stride_h * self.stride_w * self.stride_d == 1) \
                and (self.k_h * self.k_w * self.k_d == 1):
            return al1_m_data

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
        if not strideh_opti_flag:
            li_hi = self.k_h + (l1_ho - 1) * self.stride_h
        else:
            li_hi = self.k_h + (l1_ho - 1)

        return li_hi * w_i

    def _check_tiling_match(self, tiling, current_w):
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

        # get M axis length in al1
        al1_bound = self._get_al1_bound(tiling, current_w)

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
                self.k_w * self.k_d / tiling['block_dim'][1]
        else:
            # fmap size in L1 ( K * N * db * 2byte)
            filter_l1_size = utils.FP16_SIZE * tiling['BL1_shape'][1] * \
                tiling['CL0_matrix'][0] * utils.FP16_N * tiling['BL1_shape'][0] * \
                utils.FP16_K * self.k_h * self.k_w * \
                tiling['manual_pingpong_buffer']['BL1_pbuffer']

        # check exceed L1 buffer
        if int(fmap_l1_size) + int(filter_l1_size) > utils.L1BUFFER:
            return False

        # get m axis length in cub
        fused_num = self.var_map.get("fused_num", 0)
        tensor_num = fused_num + 2 if fused_num > 0 else 1
        cub_bound = (tiling['CL0_matrix'][0] * tiling['CL0_matrix'][1] * utils.FP16_M * BIT_RATIO_DICT.get(self.c_type))

        return cub_bound * tensor_num <= tbe_platform_info.get_soc_spec("UB_SIZE")

    def _check_tiling_match_d(self, current_size):
        """

        check whether this tiling matches the shape d

        Parameters
        ----------
        current_size : int, size of [d, h, w]

        Returns
        -------
        bool, True: match
            False: do not match
        """

        pad_front, pad_back, *_ = self._calc_pads(*current_size)

        return pad_front == 0 and pad_back == 0

    def get_batch_range(self, paras):
        """
        get batch covering range
        """
        batch = paras
        if "batch_n" in self.var_map:
            core_num = tbe_platform_info.get_soc_spec("CORE_NUM")
            if batch >= core_num:
                return core_num, -1
            if core_num == N_BASE:
                return 1, -1
            batch_log = int(math.log(batch, N_BASE))
            return N_BASE ** batch_log, N_BASE ** (int(batch_log + 1))
        return batch, batch

    def get_tiling_range(self, tiling_in, shape_info):
        """
        get the covered area of a tiling

        Parameters
        ----------
        tiling_in : dict, result of tiling fetch

        shape_info : list, size of fmap_shape

        Returns
        -------
        list, range covered for tiling_in
        """

        tiling = self._preprocess_tiling(tiling_in)
        fmap_n, fmap_d, _, fmap_h, fmap_w, _ = shape_info
        if not tiling["AL1_shape"]:
            # fully load in AL1, covering lower region
            return [1, fmap_n, 1, fmap_d, 1, fmap_h, 1, fmap_w]
        n_range_min, n_range_max = self.get_batch_range(fmap_n)
        perf_range = [n_range_min, n_range_max]
        # get min value
        hi_min, wi_min = 1, 1
        if self.pad_mode != "VAR":
            hi_min = max(self.k_h - self.padu - self.padd, 1)
            wi_min = max(self.k_w - self.padl - self.padr, 1)
        support_w_min = wi_min

        hi_max = H_RANGE
        cur_w_size = fmap_w

        # check tiling covering itself situation
        if not self._check_tiling_match(tiling, cur_w_size) or \
                fmap_h > H_RANGE or fmap_w > W_RANGE:
            return [0, 0, 0, 0, 0, 0]

        # searching down-ward for w_min
        while self._check_tiling_match(tiling, cur_w_size) \
                and cur_w_size > support_w_min:
            wi_min = cur_w_size
            cur_w_size = cur_w_size - W_DELTA

        # searching up-ward for w_max
        cur_w_size = fmap_w
        while self._check_tiling_match(tiling, cur_w_size) \
                and cur_w_size <= W_RANGE:
            wi_max = cur_w_size
            cur_w_size = cur_w_size + W_DELTA

        perf_di_min = 1
        perf_di_max = -1

        # If the pad in the d direction is not 0, the k value of al0_matrix can only be a factor of C1HkWk
        if tiling.get('AL0_matrix') and tiling.get('AL0_matrix')[-1] != 1:
            cur_d_size = fmap_d
            # searching down-ward for d_min
            while self._check_tiling_match_d([cur_d_size, fmap_h, fmap_w]):
                perf_di_min = cur_d_size
                cur_d_size = cur_d_size - D_DELTA

            # searching up-ward for d_max
            cur_d_size = fmap_d
            while self._check_tiling_match_d([cur_d_size, fmap_h, fmap_w]):
                perf_di_max = cur_d_size
                cur_d_size = cur_d_size + D_DELTA

        perf_wi_min = max(wi_min, fmap_w - W_LEN)
        perf_wi_max = min(wi_max, fmap_w + W_LEN)
        perf_hi_max = min(hi_max, fmap_h + H_LEN)
        perf_hi_min = max(hi_min, fmap_h - H_LEN)

        if perf_wi_min > perf_wi_max:
            return [0, 0, 0, 0, 0, 0, 0, 0]

        perf_range += [perf_di_min, perf_di_max, perf_hi_min, perf_hi_max, perf_wi_min, perf_wi_max]
        perf_range = [int(v) for v in perf_range]
        return perf_range
