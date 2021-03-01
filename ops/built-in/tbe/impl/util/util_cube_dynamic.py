# Copyright 2020 Huawei Technologies Co., Ltd
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
public function for cube dynamic
"""

from __future__ import absolute_import
import math
import copy

import te
from te.tvm import api as tvm
import te.platform as tbe_platform
from te.utils import para_check
from te.utils.error_manager import error_manager_cube as err_man
import te.lang.base as tbe_base
import impl.util.util_deconv_comm as comm

N_DIM = 0
C_DIM = 1
H_DIM = 2
W_DIM = 3
H_DIM_2D = 0
W_DIM_2D = 1
RANGE_DIM_LEN = 2
FORMAT_HW_DIM = 2
FORMAT_NCHW_DIM = 4
FORMAT_NC1HWC0_DIM = 5
DYNAMIC_FLAG = -1
UNKNOWN_FLAG = -2
DIM_TO_NAME = {0: "N", 2: "H", 3: "W"}
INPUT_SIZE_DEFAULT_SHAPE = [4]


def ceil_div(x_1, x_2):
    """
    ceil divide for inputs
    """

    if x_1 is None:
        return x_1
    if x_2 == 0:
        err_man.raise_err_specific("conv2d", "division by zero")
    return (x_1 + x_2 - 1) // x_2


def align(x_1, x_2):
    """
    align up for inputs
    """

    return ceil_div(x_1, x_2) * x_2


def lcm(x_1, x_2):
    """
    get the least common multiple
    """

    return (x_1 * x_2) // math.gcd(x_1, x_2)


def pos_from_format(ele_format):
    """
    get value from ele_format
    """
    pos_n = ele_format.find('N')
    pos_c = ele_format.find('C')
    pos_h = ele_format.find('H')
    pos_w = ele_format.find('W')
    return pos_n, pos_c, pos_h, pos_w


def set_default_para():
    """
    set default parameter value
    """
    default_para = {}
    default_para["res_dtype"] = "float16"
    default_para["input_size"] = {"ori_shape": INPUT_SIZE_DEFAULT_SHAPE}
    return default_para


class CubeParaProcess:
    """
    class of param check and preprocess for dynamic cube ops
    """

    def __init__(self, paras):
        self.paras = paras
        self.groups = paras.get("groups")
        self.strides = paras.get("strides")
        self.pads = paras.get("pads")
        self.dilations = paras.get("dilations")
        self.op_type = None
        self.valid_paras = {
            "nhw_min": 1,
            "hw_max": 4096,
            "valid_format": {"weights": ("NCHW", "NHWC", "HWCN"),
                             "input": ("NCHW", "NHWC"),
                             "output": ("NCHW", "NHWC")},
            "valid_dtype": ("float16",)
        }

    def check_support_valid(self, in_shape, filter_shape):
        """
        check whether dynamic shape is supported for cube ops
        """

        if self.groups != 1 and self.op_type != "conv2d":
            err_man.raise_err_specific_user(
                self.op_type, "group != 1 is not supported yet in dynamic")
        if DYNAMIC_FLAG not in in_shape:
            err_man.raise_err_specific_user(
                self.op_type, "need at least one dimension is a variable.")
        if DYNAMIC_FLAG in filter_shape:
            err_man.raise_err_specific_user(
                self.op_type, "dynamic weight is not supported yet.")
        if self.dilations[H_DIM] != 1 or self.dilations[W_DIM] != 1:
            err_man.raise_err_specific_user(
                self.op_type, "dilations is not supported in dynamic shape yet.")

    def check_range_valid(self, shape, dyn_range, name, in_format):
        """
        check if the range is valid
        """

        def _check_range(in_range, dim):
            dim_valid_dict = {
                N_DIM: (self.valid_paras.get("nhw_min"), None),
                H_DIM: (self.valid_paras.get("nhw_min"), self.valid_paras.get("hw_max")),
                W_DIM: (self.valid_paras.get("nhw_min"), self.valid_paras.get("hw_max"))
            }
            if in_range:
                if not isinstance(in_range, (tuple, list)):
                    err_man.raise_err_specific_user(self.op_type, "type of range must be tuple or list.")
                valid_lower, valid_upper = dim_valid_dict.get(dim)
                if len(in_range) != RANGE_DIM_LEN:
                    err_man.raise_err_specific_user(self.op_type, "each dimension of range must be 2.")
                if not (isinstance(in_range[0], int) and isinstance(in_range[1], int)):
                    err_man.raise_err_specific_user(self.op_type, "each dimension of range must be int.")
                if not in_range[0] or in_range[0] < valid_lower:
                    err_man.raise_err_attr_range_invalid(
                        self.op_type, [valid_lower, valid_upper], DIM_TO_NAME[dim] + " of " + name, in_range[0])
                if in_range[1]:
                    if valid_upper and in_range[1] > valid_upper:
                        err_man.raise_err_attr_range_invalid(
                            self.op_type, [valid_lower, valid_upper], DIM_TO_NAME[dim] + " of " + name, in_range[1])
                    if in_range[0] > in_range[1]:
                        err_man.raise_err_specific_user(self.op_type, "upper bound must be greater than lower bound.")

        if shape[C_DIM] == DYNAMIC_FLAG:
            err_man.raise_err_specific_user(self.op_type, "dynamic c dimension is not supported yet.")
        for index, dim in enumerate(zip(shape, dyn_range)):
            if dim[0] == DYNAMIC_FLAG:
                if not dim[1]:
                    err_man.raise_err_specific_user(self.op_type, "must specify range when shape is -1")
                if dim[1][1]:
                    _check_range(dim[1], index)

    def check_para_dim(self, seq, seq_name):
        """
        check if the sequence is four-dimensional
        """

        if len(seq) != FORMAT_NCHW_DIM:
            err_man.raise_err_should_be_4d(self.op_type, seq_name)

    def check_format(self, param_format, param_name):
        """
        check if the format is valid
        """

        expect_formats = self.valid_paras.get("valid_format").get(param_name)
        if param_format not in expect_formats:
            err_man.raise_err_input_format_invalid(
                self.op_type, param_name, expect_formats, param_format)

    def check_input_dict(self, para, para_name, need_range):
        """
        check if the input dict is valid
        """

        if not isinstance(para, dict):
            err_man.raise_err_check_type(self.op_type, para_name, dict, type(para))
        if not para.get("ori_shape"):
            err_man.raise_err_specific_user(self.op_type, f"need to pass ori_shape in {para_name}")
        if len(para.get("ori_shape")) != FORMAT_NCHW_DIM:
            err_man.raise_err_specific_user(self.op_type, "dim of fmap/out_backprop should be 4")
        for i in range(len(para.get("ori_shape"))):
            if not isinstance(para.get("ori_shape")[i], int):
                err_man.raise_err_specific_user(self.op_type, "value of shape must be int")
            if para.get("ori_shape")[i] <= 0 and para.get("ori_shape")[i] != DYNAMIC_FLAG:
                err_man.raise_err_specific_user(self.op_type, "value of shape must be -1 or >0")
        if not para.get("dtype"):
            err_man.raise_err_specific_user(self.op_type, f"need to pass dtype in {para_name}")
        if not para.get("ori_format"):
            err_man.raise_err_specific_user(self.op_type, f"need to pass ori_format in {para_name}")
        if need_range and not para.get("range"):
            err_man.raise_err_specific_user(self.op_type, f"need to pass range in {para_name}")

    def get_input_nchw(self, in_shape, in_format, in_range=()):
        """
        get input shape and range of nchw format
        """

        pos_n, pos_c, pos_h, pos_w = pos_from_format(in_format)
        in_shape = [in_shape[pos_n], in_shape[pos_c], in_shape[pos_h], in_shape[pos_w]]
        if in_range:
            if len(in_range) == FORMAT_NCHW_DIM:
                in_range = [in_range[pos_n], in_range[pos_c], in_range[pos_h], in_range[pos_w]]
            # range in NC1HWC0 format sometimes
            elif len(in_range) == FORMAT_NC1HWC0_DIM:
                in_range = [in_range[N_DIM], (in_shape[C_DIM], in_shape[C_DIM]), in_range[H_DIM], in_range[W_DIM]]
            else:
                err_man.raise_err_specific_user(self.op_type, "dimension of range should be 4 or 5.")
            for r in in_range:
                if not isinstance(r, (tuple, list)):
                    err_man.raise_err_specific_user(self.op_type, "each dim of range must be tuple or list.")
            return in_shape, [tuple(r) if r else r for r in in_range]
        return in_shape

    def get_attr_nchw(self, in_format):
        """
        get the input shape of nchw format
        """

        pos_n, pos_c, pos_h, pos_w = pos_from_format(in_format)
        self.dilations = [self.dilations[pos_n], self.dilations[pos_c],
                          self.dilations[pos_h], self.dilations[pos_w]]
        self.strides = [self.strides[pos_n], self.strides[pos_c],
                        self.strides[pos_h], self.strides[pos_w]]

    def get_output_range(self, w_shape, in_range, out_range=()):
        """
        calculate output range
        """

        def _get_output(x_in, k_size, pads, stride, dilation):
            if not x_in:
                return x_in
            return (x_in + pads[0] + pads[1] - dilation * (k_size - 1) - 1) // stride + 1

        if DYNAMIC_FLAG in self.pads:
            out_h_lower = ceil_div(in_range[H_DIM][0], self.strides[H_DIM])
            out_h_upper = ceil_div(in_range[H_DIM][1], self.strides[H_DIM])
            out_w_lower = ceil_div(in_range[W_DIM][0], self.strides[W_DIM])
            out_w_upper = ceil_div(in_range[W_DIM][1], self.strides[W_DIM])
        else:
            out_h_lower = _get_output(in_range[H_DIM][0], w_shape[H_DIM],
                                      (self.pads[0], self.pads[1]), self.strides[H_DIM],
                                      self.dilations[H_DIM])
            out_h_upper = _get_output(in_range[H_DIM][1], w_shape[H_DIM],
                                      (self.pads[0], self.pads[1]), self.strides[H_DIM],
                                      self.dilations[H_DIM])
            out_w_lower = _get_output(in_range[W_DIM][0], w_shape[W_DIM],
                                      (self.pads[2], self.pads[3]), self.strides[W_DIM],
                                      self.dilations[W_DIM])
            out_w_upper = _get_output(in_range[W_DIM][1], w_shape[W_DIM],
                                      (self.pads[2], self.pads[3]), self.strides[W_DIM],
                                      self.dilations[W_DIM])
        if out_h_lower <= 0 or out_w_lower <= 0:
            err_man.raise_err_specific_user(self.op_type, "lower output_range of h/w must be more than 0!")
        if out_range:
            return [out_range[N_DIM], out_range[C_DIM], (out_h_lower, out_h_upper), (out_w_lower, out_w_upper)]
        return [in_range[N_DIM], (w_shape[N_DIM], w_shape[N_DIM]),
                (out_h_lower, out_h_upper), (out_w_lower, out_w_upper)]

    def check_pads(self, dy_shape, op_type):
        """
        check pad
        """

        if op_type == "deconvolution":
            if DYNAMIC_FLAG in self.pads:
                err_man.raise_err_specific_user(self.op_type,"not support -1 in pads for deconvolution.")
            if self.pads[0]!=self.pads[1] or self.pads[2]!=self.pads[3]:
                err_man.raise_err_specific_user(self.op_type,"value of pads for deconvolution should be [A, A, B, B].")
        elif DYNAMIC_FLAG in dy_shape[1:] and (DYNAMIC_FLAG not in self.pads and sum(self.pads) != 0):
            err_man.raise_err_specific_user(self.op_type,"pads is [-1,-1,-1,-1] or [0,0,0,0] when h or w dim is -1.")

    def calc_pads(self, in_shape_nc1hwc0, w_shape):
        """
        calculate pads
        """

        pads = self.pads
        if DYNAMIC_FLAG in self.pads:
            # if load2d, return [0,0,0,0]
            if (self.op_type == "conv2d" and w_shape[H_DIM] * w_shape[W_DIM] == 1
                    and self.strides[H_DIM] * self.strides[W_DIM] == 1):
                pads = [0, 0, 0, 0]
            else:
                filter_h_dilation = (w_shape[H_DIM] - 1) * self.dilations[H_DIM] + 1
                filter_w_dilation = (w_shape[W_DIM] - 1) * self.dilations[W_DIM] + 1
                pad_h = (align(in_shape_nc1hwc0[H_DIM], self.strides[H_DIM]) -
                         self.strides[H_DIM] + filter_h_dilation - in_shape_nc1hwc0[H_DIM])
                pad_h = tvm.max(pad_h, 0)
                pad_up = pad_h // 2
                pad_down = pad_h - pad_up
                pad_w = (align(in_shape_nc1hwc0[W_DIM], self.strides[W_DIM]) -
                         self.strides[W_DIM] + filter_w_dilation - in_shape_nc1hwc0[W_DIM])
                pad_w = tvm.max(pad_w, 0)
                pad_left = pad_w // 2
                pad_right = pad_w - pad_left
                pads = pad_up, pad_down, pad_left, pad_right
                pads = list(map(lambda x: int(x) if (isinstance(x, te.tvm.expr.IntImm)) else x, pads))
        self.pads = pads

    def round_channel(self, in_shape, w_shape, dtype, out_shape=()):
        """
        round up the channel dimension
        """

        if (self.op_type == "conv2d_backprop_input" and in_shape[C_DIM] != w_shape[N_DIM]
                and out_shape[C_DIM] != w_shape[C_DIM]):
            err_man.raise_err_scene_equal_limitation(self.op_type, "input feature map channel", "filter channel")

        block_size_k, block_size_n = tbe_platform.CUBE_MKN[dtype]['mac'][1:3]
        in_shape[C_DIM] = align(in_shape[C_DIM], block_size_k)
        w_shape[N_DIM] = align(w_shape[N_DIM], block_size_n)
        w_shape[C_DIM] = align(w_shape[C_DIM], block_size_k)
        if out_shape:
            out_shape[C_DIM] = align(out_shape[C_DIM], block_size_k)

        return in_shape, w_shape, out_shape

    def set_group_para(self, in_shape, w_shape, w_dtype):
        """
        calculate paras for group
        """

        block_size_k, block_size_n = tbe_platform.CUBE_MKN[w_dtype]['mac'][1:3]
        cin_ori = in_shape[C_DIM] // self.groups
        cout_ori = w_shape[N_DIM] // self.groups
        cin_lcm = lcm(cin_ori, block_size_k) // cin_ori
        cout_lcm = lcm(cout_ori, block_size_n) // cout_ori
        enlarge = min(lcm(cin_lcm, cout_lcm), self.groups)
        c1_opt = math.ceil(cin_ori * enlarge / block_size_k)
        cout1_opt = math.ceil(cout_ori * enlarge / block_size_n)
        group_opt = math.ceil(self.groups / enlarge)

        return {"enlarge": enlarge, "c1_opt": c1_opt, "cout1_opt": cout1_opt, "group_opt": group_opt}


class Conv2dParaProcess(CubeParaProcess):
    """
    class of param check and preprocess for dynamic conv2d
    """

    def __init__(self, paras):
        super().__init__(paras)
        self.op_type = "conv2d"
        self.inputs = paras.get("inputs")
        self.weights = paras.get("weights")
        self.bias = paras.get("bias")
        self.outputs = paras.get("outputs")
        self.data_format = paras.get("data_format")
        self.dtype = paras.get("inputs").get("dtype")
        self.pooling_mode = None

    def check_support_valid(self, in_shape, w_shape):
        """
        check whether dynamic shape is supported for conv2d
        """

        super().check_support_valid(in_shape, w_shape)
        soc_version = tbe_platform.get_soc_spec("SOC_VERSION")
        if soc_version in ("Hi3796CV300ES", "Hi3796CV300CS"):
            err_man.raise_err_specific_user(
                self.op_type, "Hi3796CV300ES and Hi3796CV300CS don't support dynamic shape")
        if self.paras.get("offset_w"):
            err_man.raise_err_specific_user(
                self.op_type, "offset_w is not supported in dynamic shape yet.")

    def _calc_shape(self, in_shape, w_shape, in_range, y_range, group_para):
        """
        calculate shape for mmad
        """

        in_shape, w_shape, _ = self.round_channel(in_shape, w_shape, self.dtype)
        block_size_k, block_size_n = tbe_platform.CUBE_MKN[self.dtype]['mac'][1:3]
        # filter channel should be equal input channel
        w_shape[1] = in_shape[1]

        in_shape_nc1hwc0 = [in_shape[N_DIM], in_shape[C_DIM] // block_size_k,
                            in_shape[H_DIM], in_shape[W_DIM], block_size_k]
        if in_shape_nc1hwc0[N_DIM] == DYNAMIC_FLAG:
            in_shape_nc1hwc0[N_DIM] = tbe_base.var("batch_n", in_range[N_DIM])
            tbe_base.add_exclude_bound_var(in_shape_nc1hwc0[N_DIM])
        if in_shape_nc1hwc0[H_DIM] == DYNAMIC_FLAG:
            in_shape_nc1hwc0[H_DIM] = tbe_base.var("fmap_h", in_range[H_DIM])
            tbe_base.add_exclude_bound_var(in_shape_nc1hwc0[H_DIM])
            tbe_base.add_exclude_bound_var(tbe_base.var("ho", y_range[H_DIM]))
        if in_shape_nc1hwc0[W_DIM] == DYNAMIC_FLAG:
            in_shape_nc1hwc0[W_DIM] = tbe_base.var("fmap_w", in_range[W_DIM])
            tbe_base.add_exclude_bound_var(in_shape_nc1hwc0[W_DIM])
            tbe_base.add_exclude_bound_var(tbe_base.var("wo", y_range[W_DIM]))

        if self.paras.get("optim_dict").get("c0_optim_flg"):
            w_shape_frac_z = (ceil_div(4 * w_shape[H_DIM] * w_shape[W_DIM], block_size_k),
                              w_shape[N_DIM] // block_size_n, block_size_n, block_size_k)
        elif self.pooling_mode == "AVG":
            w_shape_frac_z = (group_para["group_opt"] * group_para["c1_opt"] * w_shape[H_DIM] * w_shape[W_DIM],
                              group_para["cout1_opt"], block_size_n, block_size_k)
        else:
            w_shape_frac_z = (group_para.get("group_opt") * group_para.get("c1_opt") * w_shape[H_DIM] * w_shape[W_DIM],
                              w_shape[N_DIM] // block_size_n, block_size_n, block_size_k)
        return in_shape, w_shape, in_shape_nc1hwc0, w_shape_frac_z

    def check_paras(self):
        """
        check original paras
        """

        padding_mode = None
        if isinstance(self.pads, str):
            self.pooling_mode = "AVG"
            padding_mode = self.pads
            if padding_mode == "VALID":
                self.pads = [0, 0, 0, 0]
            else:
                self.pads = [DYNAMIC_FLAG] * 4
        self.check_input_dict(self.inputs, "inputs", True)
        self.check_input_dict(self.weights, "weights", False)
        para_check.check_dtype_rule(self.dtype, self.valid_paras.get("valid_dtype"))
        para_check.check_dtype_rule(self.weights.get("dtype"), self.valid_paras.get("valid_dtype"))
        para_check.check_dtype_rule(self.paras.get("outputs").get("dtype"), self.valid_paras.get("valid_dtype"))
        if self.dtype != self.weights.get("dtype"):
            err_man.raise_err_specific_user("conv2d", "in_dtype != w_dtype")
        self.check_format(self.data_format, "input")
        self.check_format(self.weights.get("ori_format"), "weights")
        if self.inputs.get("ori_format") != self.data_format:
            err_man.raise_err_specific_user("conv2d", "in_format != data_format")
        para_check.check_kernel_name(self.paras.get("kernel_name"))

        in_shape = list(self.inputs.get("ori_shape"))
        in_range = self.inputs.get("range")
        w_shape = list(self.weights.get("ori_shape"))
        outputs_shape = list(self.outputs.get("ori_shape"))
        self.check_para_dim(w_shape, "weights")
        self.check_para_dim(self.strides, "strides")
        self.check_para_dim(self.dilations, "dilations")
        self.check_para_dim(self.pads, "pads")
        w_shape_nchw = self.get_input_nchw(w_shape, self.weights.get("ori_format"))
        out_shape_nchw = self.get_input_nchw(outputs_shape, self.outputs.get("ori_format"))

        if in_shape == [-2] and (outputs_shape == [DYNAMIC_FLAG, w_shape_nchw[N_DIM], DYNAMIC_FLAG, DYNAMIC_FLAG] or
                                 outputs_shape == [DYNAMIC_FLAG, DYNAMIC_FLAG, DYNAMIC_FLAG, w_shape_nchw[N_DIM]]):
            in_shape_nchw = [DYNAMIC_FLAG, w_shape_nchw[C_DIM], DYNAMIC_FLAG, DYNAMIC_FLAG]
            in_range_nchw = [(1, None), (w_shape_nchw[C_DIM], w_shape_nchw[C_DIM]), (1, None), (1, None)]
        else:
            self.check_para_dim(in_shape, "in_shape")
            in_shape_nchw, in_range_nchw = self.get_input_nchw(in_shape, self.data_format, in_range)
            self.check_range_valid(in_shape_nchw, in_range_nchw, "fmap", self.data_format)

        self.check_support_valid(in_shape_nchw, w_shape_nchw)
        self.get_attr_nchw(self.data_format)
        y_range = self.get_output_range(w_shape_nchw, in_range_nchw)
        self.check_range_valid(out_shape_nchw, y_range, "output", self.data_format)

        group_para = self.set_group_para(in_shape_nchw, w_shape_nchw, self.dtype)
        in_shape_nchw, w_shape_nchw, in_shape_nc1hwc0, w_shape_frac_z = self._calc_shape(
            in_shape_nchw, w_shape_nchw, in_range_nchw, y_range, group_para)
        self.calc_pads(in_shape_nc1hwc0, w_shape_nchw)

        return {"in_shape_nc1hwc0": in_shape_nc1hwc0, "w_shape_frac_z": w_shape_frac_z,
                "w_shape": w_shape_nchw, "group_para": group_para, "padding_mode": padding_mode,
                "pooling_mode": self.pooling_mode}

    def config_paras(self):
        """
        config paras and placeholders
        """

        param = self.check_paras()
        input_tensor = tvm.placeholder(param.get("in_shape_nc1hwc0"), name="Fmap", dtype=self.dtype)
        weight_tensor = tvm.placeholder(param.get("w_shape_frac_z"), name="Filter", dtype=self.dtype)
        if self.bias:
            bias_tensor = tvm.placeholder((param.get("w_shape")[N_DIM],), name="bias_tensor", dtype=self.dtype)
        else:
            bias_tensor = None

        return {"input_tensor": input_tensor, "weight_tensor": weight_tensor, "bias_tensor": bias_tensor,
                "w_shape": param.get("w_shape"), "in_shape_nc1hwc0": param.get("in_shape_nc1hwc0"),
                "w_shape_frac_z": param.get("w_shape_frac_z"), "group_para": param.get("group_para"),
                "padding_mode": param.get("padding_mode"), "pooling_mode": param.get("pooling_mode")}

class Conv2dBackpropParaProcess(CubeParaProcess):
    """
    class of param check and preprocess for dynamic conv2d
    """

    def __init__(self, paras):
        super().__init__(paras)
        self.op_type = "conv2d_backprop_input"
        self.filters = paras.get("filters")
        self.out_backprop = paras.get("out_backprop")
        self.y = paras.get("y")
        self.data_format = paras.get("data_format")
        self.dtype = paras.get("filters").get("dtype")
        if paras.get("input_size"):
            self.input_size = paras.get("input_size")
        else:
            self.input_size = {"ori_shape": INPUT_SIZE_DEFAULT_SHAPE}

    def _calc_shape(self, dy_shape, filter_shape, input_size, dy_range, input_range):
        """
        calculate shape for mmad
        """

        dy_shape, filter_shape, input_size = self.round_channel(dy_shape, filter_shape, self.dtype, input_size)
        block_size_k, block_size_n = tbe_platform.CUBE_MKN[self.dtype]['mac'][1:3]

        dy_shape_nc1hwc0 = [dy_shape[N_DIM], dy_shape[C_DIM] // block_size_k,
                            dy_shape[H_DIM], dy_shape[W_DIM], block_size_k]

        if input_size[N_DIM] == DYNAMIC_FLAG:
            dy_shape_nc1hwc0[N_DIM] = tbe_base.var("batch_n", dy_range[N_DIM])
            input_size[N_DIM] = dy_shape_nc1hwc0[N_DIM]
            tbe_base.add_exclude_bound_var(dy_shape_nc1hwc0[N_DIM])
        if input_size[H_DIM] == DYNAMIC_FLAG:
            dy_shape_nc1hwc0[H_DIM] = tbe_base.var("dedy_h", dy_range[H_DIM])
            input_size[H_DIM] = tbe_base.var("dx_h", input_range[H_DIM])
            tbe_base.add_exclude_bound_var(dy_shape_nc1hwc0[H_DIM])
            tbe_base.add_exclude_bound_var(input_size[H_DIM])
        if input_size[W_DIM] == DYNAMIC_FLAG:
            dy_shape_nc1hwc0[W_DIM] = tbe_base.var("dedy_w", dy_range[W_DIM])
            input_size[W_DIM] = tbe_base.var("dx_w", input_range[W_DIM])
            tbe_base.add_exclude_bound_var(dy_shape_nc1hwc0[W_DIM])
            tbe_base.add_exclude_bound_var(input_size[W_DIM])

        filter_shape_frac_z = (filter_shape[C_DIM] * filter_shape[H_DIM] * filter_shape[W_DIM] // block_size_k,
                              filter_shape[N_DIM] // block_size_n, block_size_n, block_size_k)
        return dy_shape, filter_shape, input_size, dy_shape_nc1hwc0, filter_shape_frac_z

    def _get_dy_range_dilate(self, dy_range, filter_shape):
        """
        get dy range after dilated
        """
        dy_range_dilate = dy_range
        dy_range_dilate[H_DIM] = [x * self.strides[H_DIM] for x in dy_range[H_DIM]]
        if filter_shape[H_DIM] == 1 and filter_shape[W_DIM] == 1:
            dy_range_dilate[W_DIM] = [x * self.strides[H_DIM] * self.strides[W_DIM] for x in dy_range[W_DIM]]
        else:
            dy_range_dilate[W_DIM] = [x * self.strides[W_DIM] for x in dy_range[W_DIM]]
        return dy_range_dilate

    def infer_shape_and_range(self):
        """
        infer range from dx to dy
        """
        self.check_input_dict(self.y, "y", True)

        dy_shape = self.out_backprop.get("ori_shape")
        filter_shape = self.filters.get("ori_shape")
        dx_shape = self.y.get("ori_shape")
        dx_range = self.y.get("range")
        self.check_para_dim(dx_shape, "input_size")
        self.check_para_dim(filter_shape, "filters")
        self.check_pads(dy_shape, self.op_type)
        filter_shape_nchw = self.get_input_nchw(filter_shape, self.filters.get("ori_format"))
        self.get_attr_nchw(self.data_format)
        dx_shape_nchw, dx_range_nchw = self.get_input_nchw(dx_shape, self.data_format, dx_range)

        if dy_shape == [-2]:
            dy_shape_nchw = [DYNAMIC_FLAG, filter_shape_nchw[N_DIM], DYNAMIC_FLAG, DYNAMIC_FLAG]
            dy_range_nchw = [(1, None), (filter_shape_nchw[N_DIM], filter_shape_nchw[N_DIM]), (1, None), (1, None)]
            self.check_support_valid(dy_shape_nchw, filter_shape_nchw)
            group_para = comm.calculate_group(
                dy_shape_nchw, dx_shape_nchw, filter_shape_nchw, self.groups, self.dtype, self.data_format)
            comm.check_conv2dbp_input_params(filter_shape_nchw, dy_shape_nchw, dx_shape_nchw,
                                             self.strides[2:], self.pads, self.dilations, self.dtype, self.dtype, self.dtype,
                                             kernel_name=self.paras.get("kernel_name"),
                                             fusion_para=None,
                                             group_dict=group_para)
            
        else:
            self.check_para_dim(dy_shape, "out_backprop_shape")
            dy_shape_nchw = self.get_input_nchw(dy_shape, self.data_format)
            self.check_support_valid(dy_shape_nchw, filter_shape_nchw)
            group_para = comm.calculate_group(
                dy_shape_nchw, dx_shape_nchw, filter_shape_nchw, self.groups, self.dtype, self.data_format)
            comm.check_conv2dbp_input_params(filter_shape_nchw, dy_shape_nchw, dx_shape_nchw,
                                             self.strides[2:], self.pads, self.dilations, self.dtype, self.dtype, self.dtype,
                                             kernel_name=self.paras.get("kernel_name"),
                                             fusion_para=None,
                                             group_dict=group_para)
            self.check_range_valid(dx_shape_nchw, dx_range_nchw, "input_size", self.data_format)

            dy_range_nchw = self.get_output_range(filter_shape_nchw, dx_range_nchw)


        output_range = copy.deepcopy(dy_range_nchw)
        if output_range[H_DIM][1]:
            output_range[H_DIM] = (output_range[H_DIM][0], output_range[H_DIM][1] * self.strides[H_DIM])
        if output_range[W_DIM][1]:
            if filter_shape_nchw[H_DIM] == 1 and filter_shape_nchw[W_DIM] == 1:
                output_range[W_DIM] = (output_range[W_DIM][0],
                                       output_range[W_DIM][1] * self.strides[H_DIM] * self.strides[W_DIM])
            else:
                output_range[W_DIM] = (output_range[W_DIM][0], output_range[W_DIM][1] * self.strides[W_DIM])
        self.check_range_valid(dy_shape_nchw, output_range, "out_backprop", self.data_format)

        return dy_shape_nchw, filter_shape_nchw, dx_shape_nchw, dy_range_nchw, dx_range_nchw, group_para

    def check_paras(self):
        """
        check original paras
        """
        self.check_input_dict(self.filters, "filters", False)
        self.check_input_dict(self.out_backprop, "out_backprop", False)
        self.check_input_dict(self.y, "y", False)
        para_check.check_dtype_rule(self.dtype, self.valid_paras.get("valid_dtype"))
        if UNKNOWN_FLAG in self.input_size.get("ori_shape") or DYNAMIC_FLAG in self.input_size.get("ori_shape"):
            err_man.raise_err_specific_user(
                self.op_type, "dynamic shape nout support input size's shape [-1] and [-2]")
        if self.dtype != self.y.get("dtype"):
            err_man.raise_err_specific_user(
                "conv2d_backprop_input", "the dtype of filter and y are not the same.")
        if self.dtype != self.out_backprop.get("dtype"):
            err_man.raise_err_specific_user(
                "conv2d_backprop_input", "the dtype of filter and out_backprop are not the same.")

        self.check_format(self.data_format, "output")
        self.check_format(self.filters.get("ori_format"), "weights")
        if self.out_backprop.get("ori_format") != self.data_format:
            err_man.raise_err_specific_user(
                "conv2d_backprop_input", "the format of out_backprop and data_format are not the same.")
        if self.y.get("ori_format") != self.data_format:
            err_man.raise_err_specific_user("the format of y and data_format are not the same.")
        para_check.check_kernel_name(self.paras.get("kernel_name"))
        self.check_para_dim(self.strides, "strides")
        self.check_para_dim(self.dilations, "dilations")
        self.check_para_dim(self.pads, "pads")

        dy_shape_nchw, filter_shape_nchw, input_size_nchw, dy_range_nchw, input_range_nchw, group_para = self.infer_shape_and_range()

        dy_shape_nchw, filter_shape_nchw, input_size_nchw, dy_shape_nc1hwc0, filter_shape_frac_z = self._calc_shape(
            dy_shape_nchw, filter_shape_nchw, input_size_nchw, dy_range_nchw, input_range_nchw)
        self.calc_pads(input_size_nchw, filter_shape_nchw)

        return {"dy_shape_nc1hwc0": dy_shape_nc1hwc0, "filter_shape_frac_z": filter_shape_frac_z,
                "filter_shape": filter_shape_nchw, "input_size": input_size_nchw, "group_para": group_para}

    def config_paras(self):
        """
        config paras and placeholders
        """
        param = self.check_paras()
        input_tensor = tvm.placeholder([4], name="input_size", dtype="int32")
        dy_tensor = tvm.placeholder(param.get("dy_shape_nc1hwc0"), name="dedy", dtype=self.dtype)
        filter_tensor = tvm.placeholder(param.get("filter_shape_frac_z"), name="filter", dtype=self.dtype)

        return {"dy_tensor": dy_tensor, "filter_tensor": filter_tensor, "input_tensor": input_tensor,
                "filter_shape": param.get("filter_shape"), "input_size": param.get("input_size"),
                "group_para": param.get("group_para")}


class Conv2dTransposeParaProcess(Conv2dBackpropParaProcess):
    """
    class of param check and preprocess for dynamic conv2d_transpose
    """

    def __init__(self, paras):
        super().__init__(paras)
        self.op_type = "conv2d_transpose"
        self.out_backprop = paras.get("x")

    def check_support_valid(self, in_shape, w_shape):
        """
        check whether dynamic shape is supported for conv2d_transpose
        """
        super().check_support_valid(in_shape, w_shape)
        if self.paras.get("offset_w"):
            err_man.raise_err_specific_user(
                self.op_type, "offset_w is not supported in dynamic shape yet.")
        if self.paras.get("bias"):
            err_man.raise_err_specific_user(
                self.op_type, "bias is not supported in dynamic shape yet.")
        if self.paras.get("output_padding") != (0, 0, 0, 0):
            err_man.raise_err_specific_user(
                self.op_type, "output_padding is not supported in dynamic shape yet.")
        if self.paras.get("offset_x") != 0:
            err_man.raise_err_specific_user(
                self.op_type, "offset_x is not supported in dynamic shape yet.")

    def get_input_range(self, w_shape, dy_range, dx_range=()):
        """
        calculate input range
        """

        def _get_lower_input(y_in, k_size, pads, stride, dilation):
            if not y_in:
                return y_in
            # dilation = 1
            return stride * (y_in - 1) + dilation * (k_size - 1) + 1 - pads[0] - pads[1]

        def _get_higher_input(y_in, k_size, pads, stride, dilation):
            if not y_in:
                return y_in
            # dilation = 1
            return stride * (y_in - 1) + dilation * (k_size - 1) + 1 - pads[0] - pads[1] + stride - 1

        if DYNAMIC_FLAG in self.pads:
            dx_h_lower = (dy_range[H_DIM][0]-1) * self.strides[H_DIM]+1
            if not dy_range[H_DIM][1]:
                dx_h_upper = dy_range[H_DIM][1]
            else:
                dx_h_upper = min(dy_range[H_DIM][1] * self.strides[H_DIM], self.valid_paras.get("hw_max"))
            dx_w_lower = (dy_range[W_DIM][0]-1) * self.strides[W_DIM]+1
            if not dy_range[W_DIM][1]:
                dx_w_upper = dy_range[W_DIM][1]
            else:
                dx_w_upper = min(dy_range[W_DIM][1] * self.strides[W_DIM], self.valid_paras.get("hw_max"))
        else:
            dx_h_lower = max(_get_lower_input(dy_range[H_DIM][0], w_shape[H_DIM],
                                      (self.pads[0], self.pads[1]), self.strides[H_DIM],
                                      self.dilations[H_DIM]), self.valid_paras.get("nhw_min"))
            if not dy_range[H_DIM][1]:
                dx_h_upper = dy_range[H_DIM][1]
            else:
                dx_h_upper = min(_get_higher_input(dy_range[H_DIM][1], w_shape[H_DIM],
                                          (self.pads[0], self.pads[1]), self.strides[H_DIM],
                                          self.dilations[H_DIM]), self.valid_paras.get("hw_max"))
            dx_w_lower = max(_get_lower_input(dy_range[W_DIM][0], w_shape[W_DIM],
                                      (self.pads[2], self.pads[3]), self.strides[W_DIM],
                                      self.dilations[W_DIM]), self.valid_paras.get("nhw_min"))
            if not dy_range[W_DIM][1]:
                dx_w_upper = dy_range[W_DIM][1]
            else:
                dx_w_upper = min(_get_higher_input(dy_range[W_DIM][1], w_shape[W_DIM],
                                          (self.pads[2], self.pads[3]), self.strides[W_DIM],
                                          self.dilations[W_DIM]), self.valid_paras.get("hw_max"))
        if dx_range:
            return [dx_range[N_DIM], dx_range[C_DIM], (dx_h_lower, dx_h_upper), (dx_w_lower, dx_w_upper)]
        return [dy_range[N_DIM], (w_shape[N_DIM], w_shape[N_DIM]),
                (dx_h_lower, dx_h_upper), (dx_w_lower,  dx_w_upper)]

    def infer_shape_and_range(self):
        """
        infer range from dy to dx
        """

        self.check_input_dict(self.out_backprop, "out_backprop", True)

        dy_shape = self.out_backprop.get("ori_shape")
        dy_range = self.out_backprop.get("range")
        filter_shape = self.filters.get("ori_shape")
        dx_shape = self.y.get("ori_shape")
        self.check_para_dim(dx_shape, "input_size")
        self.check_para_dim(filter_shape, "filters")
        self.check_pads(dy_shape, self.op_type)
        filter_shape_nchw = self.get_input_nchw(filter_shape, self.filters.get("ori_format"))
        self.get_attr_nchw(self.data_format)
        dx_shape_nchw = self.get_input_nchw(dx_shape, self.data_format)

        if dy_shape == [-2]:
            dy_shape_nchw = [DYNAMIC_FLAG, filter_shape_nchw[N_DIM], DYNAMIC_FLAG, DYNAMIC_FLAG]
            dy_range_nchw = [(1, None), (filter_shape_nchw[N_DIM], filter_shape_nchw[N_DIM]), (1, None), (1, None)]
            self.check_support_valid(dy_shape_nchw, filter_shape_nchw)
            group_para = comm.calculate_group(
                dy_shape_nchw, dx_shape_nchw, filter_shape_nchw, self.groups, self.dtype, self.data_format)
            comm.check_conv2dbp_input_params(filter_shape_nchw, dy_shape_nchw, dx_shape_nchw,
                                             self.strides[2:], self.pads, self.dilations, self.dtype, self.dtype, self.dtype,
                                             kernel_name=self.paras.get("kernel_name"),
                                             fusion_para=None,
                                             group_dict=group_para)
        else:
            self.check_para_dim(dy_shape, "out_backprop_shape") 
            dy_shape_nchw, dy_range_nchw = self.get_input_nchw(dy_shape, self.data_format, dy_range)
            output_range = copy.deepcopy(dy_range_nchw)
            self.check_support_valid(dy_shape_nchw, filter_shape_nchw)
            group_para = comm.calculate_group(
                dy_shape_nchw, dx_shape_nchw, filter_shape_nchw, self.groups, self.dtype, self.data_format)
            comm.check_conv2dbp_input_params(filter_shape_nchw, dy_shape_nchw, dx_shape_nchw,
                                             self.strides[2:], self.pads, self.dilations, self.dtype, self.dtype, self.dtype,
                                             kernel_name=self.paras.get("kernel_name"),
                                             fusion_para=None,
                                             group_dict=group_para)
            self.check_range_valid(dy_shape_nchw, output_range, "out_backprop", self.data_format)
            if output_range[H_DIM][1]:
                output_range[H_DIM] = (output_range[H_DIM][0], output_range[H_DIM][1] * self.strides[H_DIM])
            if output_range[W_DIM][1]:
                if filter_shape_nchw[H_DIM] == 1 and filter_shape_nchw[W_DIM] == 1:
                    output_range[W_DIM] = (output_range[W_DIM][0],
                                           output_range[W_DIM][1] * self.strides[H_DIM] * self.strides[W_DIM])
                else:
                    output_range[W_DIM] = (output_range[W_DIM][0], output_range[W_DIM][1] * self.strides[W_DIM])
            self.check_range_valid(dy_shape_nchw, output_range, "out_backprop", self.data_format)
            dx_range_nchw = self.get_input_range(filter_shape_nchw, dy_range_nchw)

        self.check_range_valid(dx_shape_nchw, dx_range_nchw, "input_size", self.data_format)

        return dy_shape_nchw, filter_shape_nchw, dx_shape_nchw, dy_range_nchw, dx_range_nchw, group_para


    def config_paras(self):
        """
        check original paras
        """
        param = super().check_paras()

        input_tensor = tvm.placeholder([4], name="input_size", dtype="int32")
        x_tensor = tvm.placeholder(param.get("dy_shape_nc1hwc0"), name="dedy", dtype=self.dtype)
        filter_tensor = tvm.placeholder(param.get("filter_shape_frac_z"), name="filter", dtype=self.dtype)
        if self.paras.get("bias"):
            bias_tensor = tvm.placeholder((param.get("filter_shape")[N_DIM],), name="tensor_bias", dtype=self.dtype)
        else:
            bias_tensor = None

        return {"x_tensor": x_tensor, "filter_tensor": filter_tensor, "input_tensor": input_tensor,
                "bias_tensor": bias_tensor, "filter_shape": param.get("filter_shape"),
                "input_size": param.get("input_size"), "group_para": param.get("group_para")}


class DeconvolutionParaProcess(Conv2dBackpropParaProcess):
    """
    class of param check and preprocess for dynamic deconvolution
    """

    def __init__(self, paras):
        super().__init__(paras)
        self.op_type = "deconvolution"
        self.out_backprop = paras.get("x")
        self.valid_paras = {
            "nhw_min": 1,
            "hw_max": 4096,
            "valid_format": {"weights": ("NCHW",),
                             "input": ("NCHW",),
                             "output": ("NCHW",)},
            "valid_dtype": ("float16",)
        }

    def check_support_valid(self, in_shape, w_shape):
        """
        check whether dynamic shape is supported for deconvolution
        """
        super().check_support_valid(in_shape, w_shape)
        if self.paras.get("offset_w"):
            err_man.raise_err_specific_user(
                self.op_type, "offset_w is not supported in dynamic shape yet.")
        if self.paras.get("bias"):
            err_man.raise_err_specific_user(
                self.op_type, "bias is not supported in dynamic shape yet.")
        if self.paras.get("offset_x") != 0:
            err_man.raise_err_specific_user(
                self.op_type, "offset_x is not supported in dynamic shape yet.")

    def get_input_range(self, w_shape, dy_range, dx_range=()):
        """
        calculate input range
        """

        def _get_input(y_in, k_size, pads, stride, dilation):
            if not y_in:
                return y_in
            # dilation = 1
            return stride * (y_in - 1) + dilation * (k_size - 1) + 1 - pads[0] - pads[1]

        dx_h_lower = max(_get_input(dy_range[H_DIM][0], w_shape[H_DIM],
                                  (self.pads[0], self.pads[1]), self.strides[H_DIM],
                                  self.dilations[H_DIM]), self.valid_paras.get("nhw_min"))
        if not dy_range[H_DIM][1]:
            dx_h_upper = dy_range[H_DIM][1]
        else:
            dx_h_upper = min(_get_input(dy_range[H_DIM][1], w_shape[H_DIM],
                                      (self.pads[0], self.pads[1]), self.strides[H_DIM],
                                      self.dilations[H_DIM]), self.valid_paras.get("hw_max"))
        dx_w_lower = max(_get_input(dy_range[W_DIM][0], w_shape[W_DIM],
                                  (self.pads[2], self.pads[3]), self.strides[W_DIM],
                                  self.dilations[W_DIM]), self.valid_paras.get("nhw_min"))
        if not dy_range[W_DIM][1]:
            dx_w_upper = dy_range[W_DIM][1]
        else:
            dx_w_upper = min(_get_input(dy_range[W_DIM][1], w_shape[W_DIM],
                                      (self.pads[2], self.pads[3]), self.strides[W_DIM],
                                      self.dilations[W_DIM]), self.valid_paras.get("hw_max"))
        if dx_range:
            return [dx_range[N_DIM], dx_range[C_DIM], (dx_h_lower, dx_h_upper), (dx_w_lower, dx_w_upper)]
        return [dy_range[N_DIM], (w_shape[N_DIM], w_shape[N_DIM]),
                (dx_h_lower, dx_h_upper), (dx_w_lower,  dx_w_upper)]

    def infer_shape_and_range(self):
        """
        infer range from dy to dx
        """

        self.check_input_dict(self.out_backprop, "out_backprop", True)

        dy_shape = self.out_backprop.get("ori_shape")
        dy_range = self.out_backprop.get("range")
        filter_shape = self.filters.get("ori_shape")
        dx_shape = self.y.get("ori_shape")
        self.check_para_dim(dx_shape, "input_size")
        self.check_para_dim(filter_shape, "filters")
        self.check_pads(dy_shape, self.op_type)
        filter_shape_nchw = self.get_input_nchw(filter_shape, self.filters.get("ori_format"))
        self.get_attr_nchw(self.data_format)
        dx_shape_nchw = self.get_input_nchw(dx_shape, self.data_format)

        if dy_shape == [-2]:
            dy_shape_nchw = [DYNAMIC_FLAG, filter_shape_nchw[N_DIM], DYNAMIC_FLAG, DYNAMIC_FLAG]
            dy_range_nchw = [(1, None), (filter_shape_nchw[N_DIM], filter_shape_nchw[N_DIM]), (1, None), (1, None)]
            self.check_support_valid(dy_shape_nchw, filter_shape_nchw)
            group_para = comm.calculate_group(
                dy_shape_nchw, dx_shape_nchw, filter_shape_nchw, self.groups, self.dtype, self.data_format)
            comm.check_conv2dbp_input_params(filter_shape_nchw, dy_shape_nchw, dx_shape_nchw,
                                             self.strides[2:], self.pads, self.dilations, self.dtype, self.dtype, self.dtype,
                                             kernel_name=self.paras.get("kernel_name"),
                                             fusion_para=None,
                                             group_dict=group_para)
        else:
            self.check_para_dim(dy_shape, "out_backprop_shape") 
            dy_shape_nchw, dy_range_nchw = self.get_input_nchw(dy_shape, self.data_format, dy_range)
            output_range = copy.deepcopy(dy_range_nchw)
            self.check_support_valid(dy_shape_nchw, filter_shape_nchw)
            group_para = comm.calculate_group(
                dy_shape_nchw, dx_shape_nchw, filter_shape_nchw, self.groups, self.dtype, self.data_format)
            comm.check_conv2dbp_input_params(filter_shape_nchw, dy_shape_nchw, dx_shape_nchw,
                                             self.strides[2:], self.pads, self.dilations, self.dtype, self.dtype, self.dtype,
                                             kernel_name=self.paras.get("kernel_name"),
                                             fusion_para=None,
                                             group_dict=group_para)
            self.check_range_valid(dy_shape_nchw, output_range, "out_backprop", self.data_format)
            if output_range[H_DIM][1]:
                output_range[H_DIM] = (output_range[H_DIM][0], output_range[H_DIM][1] * self.strides[H_DIM])
            if output_range[W_DIM][1]:
                if filter_shape_nchw[H_DIM] == 1 and filter_shape_nchw[W_DIM] == 1:
                    output_range[W_DIM] = (output_range[W_DIM][0],
                                           output_range[W_DIM][1] * self.strides[H_DIM] * self.strides[W_DIM])
                else:
                    output_range[W_DIM] = (output_range[W_DIM][0], output_range[W_DIM][1] * self.strides[W_DIM])
            self.check_range_valid(dy_shape_nchw, output_range, "out_backprop", self.data_format)
            dx_range_nchw = self.get_input_range(filter_shape_nchw, dy_range_nchw)

        self.check_range_valid(dx_shape_nchw, dx_range_nchw, "input_size", self.data_format)

        return dy_shape_nchw, filter_shape_nchw, dx_shape_nchw, dy_range_nchw, dx_range_nchw, group_para


    def config_paras(self):
        """
        check original paras
        """
        if len(self.strides) != FORMAT_HW_DIM:
            err_man.raise_err_specific_user(
                self.op_type, "length of stride in deconvolution should be 2.")
        self.strides = [1, 1, self.strides[H_DIM_2D], self.strides[W_DIM_2D]]

        param = super().check_paras()

        x_tensor = tvm.placeholder(param.get("dy_shape_nc1hwc0"), name="dedy", dtype=self.dtype)
        filter_tensor = tvm.placeholder(param.get("filter_shape_frac_z"), name="filter", dtype=self.dtype)
        if self.paras.get("bias"):
            bias_tensor = tvm.placeholder((param.get("filter_shape")[N_DIM],), name="tensor_bias", dtype=self.dtype)
        else:
            bias_tensor = None

        return {"x_tensor": x_tensor, "filter_tensor": filter_tensor,
                "bias_tensor": bias_tensor, "filter_shape": param.get("filter_shape"),
                "input_size": param.get("input_size"), "group_para": param.get("group_para")}
