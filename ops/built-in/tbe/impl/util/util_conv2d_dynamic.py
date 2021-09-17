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
import warnings
import math
import copy

from impl.util.platform_adapter import error_manager_cube as err_man
from impl.util.platform_adapter import operation
from impl.util.platform_adapter import para_check
from impl.util.platform_adapter import tbe
from impl.util.platform_adapter import tbe_platform
from impl.util.platform_adapter import tvm
from impl.util.util_cube_dynamic import CubeParaProcess
import impl.util.util_deconv_comm as comm
from te.platform import get_soc_spec
from te.platform import cce_params
from tbe.common.context import get_context


# the bytes length of several dtype
BIT_RATIO_DICT = {"int32": 4, "float32": 4, "float16": 2,
                  "uint8": 1, "int8": 1, "uint4": 0.5, "int4": 0.5}
N_DIM = 0
C_DIM = 1
H_DIM = 2
W_DIM = 3
DYNAMIC_FLAG = -1
SHAPE_LEN = 5
ORI_SHAPE_LEN = 4

def modify_input_range(in_range_nchw, data_type, idx_h, idx_w, strides, hk_dilation, wk_dilation, pads):
    '''
    check for not bigger than L1
    '''
    fmap_w_min = in_range_nchw[W_DIM][0]
    fmap_w_max = in_range_nchw[W_DIM][1]
    m_bit_ratio = {"float16": 2, "int8": 1}
    c0 = tbe_platform.CUBE_MKN[data_type]["mac"][1]
    fmap_w_upper = in_range_nchw[W_DIM][1]
    new_in_range_nchw = list(in_range_nchw)

    stride_h = strides[idx_h]
    stride_w = strides[idx_w]
    l1size_limit_upper = tbe_platform.get_soc_spec("L1_SIZE")
    w_left = fmap_w_min
    w_right = fmap_w_max
    current_w = fmap_w_max
    while (w_right - w_left) != 1:
        if -1 in pads:
            w_out = (current_w + stride_w - 1) // stride_w
        else:
            w_out = math.floor((current_w - wk_dilation + pads[2] + pads[3]) / stride_w) + 1
        ho_num = math.floor(tbe_platform.CUBE_MKN[data_type]["mac"][0] / w_out) + 2
        l1_m = ((ho_num - 1) * stride_h + hk_dilation) * current_w
        max_feature_map_l1 = c0 * l1_m * m_bit_ratio[data_type]
        if max_feature_map_l1 > l1size_limit_upper:
            w_right = current_w
        else:
            w_left = current_w
        current_w = w_left + (w_right - w_left)//2

        if w_left == fmap_w_max:
            break
    cor_w_range = (fmap_w_min, w_left)
    new_in_range_nchw[W_DIM] = cor_w_range
    to_print = "conv2d fmap ori_range changed from {} to {}.".format(in_range_nchw, new_in_range_nchw)
    warnings.warn(to_print)

    return new_in_range_nchw

def check_l1_size(op_type, inputs, kh_dilate, kw_dilate, strides, pads):
    """
    check exceed l1 buf
    """
    l1_size = tbe_platform.get_soc_spec("L1_SIZE")
    # default type fp16
    type_byte = BIT_RATIO_DICT.get(inputs["dtype"], 2)
    idx_h = inputs.get("ori_format").find("H")
    idx_w = inputs.get("ori_format").find("W")
    w_in = inputs.get("ori_shape")[idx_w]
    pad_top, pad_bottom, pad_left, pad_right = pads
    if DYNAMIC_FLAG in pads:
        w_out = w_in + strides[idx_w] - 1 // strides[idx_w]
    else:
        w_out = (w_in + (pad_left + pad_right) - kw_dilate) // strides[idx_w] + 1
    limit_h_out = math.floor(tbe_platform.CUBE_MKN[inputs["dtype"]]['mac'][0] / w_out) + 2
    hw_size = ((limit_h_out - 1) * strides[idx_h] + kh_dilate) * w_in
    limit_size = hw_size * tbe_platform.CUBE_MKN[inputs["dtype"]]['mac'][1] * type_byte
    if limit_size > l1_size:
        err_man.raise_err_specific_user(op_type, "input range is too large, the mininum tiling may exceed L1buffer.")


def create_fuzz_range(op_type, dim_value, grade_item):
    """
    gen fuzz range
    """
    # dim_value must less than grade max_value
    if dim_value > grade_item[-1]:
        err_man.raise_err_specific_user(op_type, "input value {} is out the range of {}", dim_value, grade_item[-1])
        err_man.raise_err_attr_range_invalid(op_type, "[1, 4096]", "input shape", str(dim_value))

    for g_value in grade_item:
        if dim_value > g_value:
            low = g_value + 1
        if dim_value <= g_value:
            high = g_value
            break
    return [low, high]

def correct_input_range(op_type, input_range, x_shape, idx_h, idx_w, kh_dilate, kw_dilate, pads):
    """
    correct range[low] for  output >= 1
    """
    padt, padd, padl, padr = pads
    if DYNAMIC_FLAG in pads:# padding=same mode
        low_h = input_range[idx_h][0]
        low_w = input_range[idx_w][0]
    else:
        low_h = kh_dilate - padt - padd
        if x_shape[idx_h] != DYNAMIC_FLAG and x_shape[idx_h] < low_h:
            err_man.raise_err_specific_user(op_type, "the output h must be greater than or equal to 1")
        if x_shape[idx_h] == DYNAMIC_FLAG and input_range[idx_h][1] < low_h:
            err_man.raise_err_specific_user(op_type, "the h_range[1] must be greater than or equal to kerne_h")
        low_w = kw_dilate - padl - padr
        if x_shape[idx_w] != DYNAMIC_FLAG and x_shape[idx_w] < low_w:
            err_man.raise_err_specific_user(op_type, "the output w must be greater than or equal to 1")
        if x_shape[idx_w] == DYNAMIC_FLAG and input_range[idx_w][1] < low_w:
            err_man.raise_err_specific_user(op_type, "the w_range[1] must be greater than or equal to kerne_w")

    input_range[idx_h][0] = max(input_range[idx_h][0], low_h)
    input_range[idx_w][0] = max(input_range[idx_w][0], low_w)


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


class Conv2dParaProcess(CubeParaProcess):
    """
    class of param check and preprocess for dynamic conv2d
    """

    def __init__(self, paras):
        def conver_tensor2dict(tensor, need_range):
            if tensor is None:
                return None
            dict = {}
            dict["ori_shape"] = []
            for i in tensor.op.attrs['ori_shape']:
                dict["ori_shape"].append(i.value)
            dict["dtype"] = tensor.dtype
            dict["ori_format"] = tensor.op.attrs['ori_format'].value

            if need_range == True:
                dict["range"] = []
                for one_range in tensor.op.attrs['range']:
                    range_list = []
                    for value in one_range:
                        range_list.append(value.value)
                    dict["range"].append(range_list)
                if operation.get_te_var("batch_n"):
                    dict["range"][N_DIM] = list(operation.get_te_var("batch_n").get_bound())
                if operation.get_te_var("fmap_h"):
                    dict["range"][H_DIM] = list(operation.get_te_var("fmap_h").get_bound())
                if operation.get_te_var("fmap_w"):
                    dict["range"][W_DIM] = list(operation.get_te_var("fmap_w").get_bound())

            return dict

        super().__init__(paras)
        self.op_type = "conv2d"
        if isinstance(paras.get("inputs"), dict):
            self.is_tensor = False
            self.inputs = paras.get("inputs")
            self.weights = paras.get("weights")
            self.bias = paras.get("bias")
            self.dtype = paras.get("inputs").get("dtype")
        else:
            self.is_tensor = True
            self.input_tensor = paras.get("inputs")
            self.weights_tensor = paras.get("weights")
            self.bias_tensor = paras.get("bias")

            self.inputs = conver_tensor2dict(self.input_tensor, True)
            self.weights = conver_tensor2dict(self.weights_tensor, False)
            self.bias = conver_tensor2dict(self.bias_tensor, False)
            self.dtype = self.input_tensor.dtype

        self.outputs = paras.get("outputs")
        self.data_format = paras.get("data_format")

    def check_support_valid(self, in_shape, w_shape):
        """
        check whether dynamic shape is supported for conv2d
        """

        super().check_support_valid(in_shape, w_shape)
        if in_shape[C_DIM] == DYNAMIC_FLAG:
            err_man.raise_err_specific_user(
                self.op_type, "dynamic c dimension is not supported yet.")
        soc_version = tbe_platform.get_soc_spec("SOC_VERSION")
        if self.paras.get("offset_w"):
            err_man.raise_err_specific_user(
                self.op_type, "offset_w is not supported in dynamic shape yet.")

    def _calc_shape(self, in_shape, w_shape, in_range, y_range, group_para):
        """
        calculate shape for mmad
        """

        block_size_k, block_size_n = tbe_platform.CUBE_MKN[self.dtype]['mac'][1:3]
        in_shape[C_DIM] = align(in_shape[C_DIM], block_size_k)
        # filter channel should be equal input channel
        w_shape[C_DIM] = in_shape[C_DIM]

        in_shape_nc1hwc0 = [in_shape[N_DIM], in_shape[C_DIM] // block_size_k,
                            in_shape[H_DIM], in_shape[W_DIM], block_size_k]
        if self.is_tensor == False:
            if in_shape_nc1hwc0[N_DIM] == DYNAMIC_FLAG:
                in_shape_nc1hwc0[N_DIM] = operation.var("batch_n", in_range[N_DIM])
                operation.add_exclude_bound_var(in_shape_nc1hwc0[N_DIM])
            if in_shape_nc1hwc0[H_DIM] == DYNAMIC_FLAG:
                in_shape_nc1hwc0[H_DIM] = operation.var("fmap_h", in_range[H_DIM])
                operation.add_exclude_bound_var(in_shape_nc1hwc0[H_DIM])
                operation.add_exclude_bound_var(operation.var("ho", y_range[H_DIM]))
            if in_shape_nc1hwc0[W_DIM] == DYNAMIC_FLAG:
                in_shape_nc1hwc0[W_DIM] = operation.var("fmap_w", in_range[W_DIM])
                operation.add_exclude_bound_var(in_shape_nc1hwc0[W_DIM])
                operation.add_exclude_bound_var(operation.var("wo", y_range[W_DIM]))
        else:
            if in_shape_nc1hwc0[N_DIM] == DYNAMIC_FLAG:
                in_shape_nc1hwc0[N_DIM] = self.input_tensor.shape[N_DIM]
            if in_shape_nc1hwc0[H_DIM] == DYNAMIC_FLAG:
                in_shape_nc1hwc0[H_DIM] = self.input_tensor.shape[H_DIM]
                operation.add_exclude_bound_var(operation.var("ho", y_range[H_DIM]))
            if in_shape_nc1hwc0[W_DIM] == DYNAMIC_FLAG:
                in_shape_nc1hwc0[W_DIM] = self.input_tensor.shape[W_DIM]
                operation.add_exclude_bound_var(operation.var("wo", y_range[W_DIM]))

        if self.paras.get("optim_dict").get("c0_optim_flg"):
            w_shape_frac_z = (ceil_div(4 * w_shape[H_DIM] * w_shape[W_DIM], block_size_k),
                              math.ceil(w_shape[N_DIM] / block_size_n), block_size_n, block_size_k)
        else:
            w_shape_frac_z = (group_para.get("group_opt") * group_para.get("c1_opt") * w_shape[H_DIM] * w_shape[W_DIM],
                              group_para.get("cout1_opt"), block_size_n, block_size_k)
        return in_shape, w_shape, in_shape_nc1hwc0, w_shape_frac_z

    def correct_in_range(self, in_range_nchw, w_shape_nchw):
        #correct in_range when w_range=[1, None] or fuzz_build mode
        DYNAMIC_FMAP_W_MIN = 1
        DYNAMIC_FMAP_W_MAX = 4096
        fuzz_build = get_context().get_build_type() == "fuzzily_build"
        fmap_w_min = in_range_nchw[W_DIM][0] if fuzz_build else DYNAMIC_FMAP_W_MIN
        fmap_w_max = in_range_nchw[W_DIM][1] if fuzz_build else DYNAMIC_FMAP_W_MAX
        m_bit_ratio = {"float16": 2, "int8": 1}
        c0 = tbe_platform.CUBE_MKN[self.weights["dtype"]]["mac"][1]
        fmap_w_upper = in_range_nchw[W_DIM][1]
        new_in_range_nchw = list(in_range_nchw)

        if not fmap_w_upper or fuzz_build:
            stride_h = self.strides[H_DIM]
            stride_w = self.strides[W_DIM]
            hk_dilation = (w_shape_nchw[H_DIM] - 1) * self.dilations[H_DIM] + 1
            wk_dilation = (w_shape_nchw[W_DIM] - 1) * self.dilations[W_DIM] + 1
            l1size_limit_upper = tbe_platform.get_soc_spec("L1_SIZE")
            w_left = fmap_w_min
            w_right = fmap_w_max
            current_w = fmap_w_max
            while (w_right - w_left) != 1:
                if -1 in self.pads:
                    w_out = (current_w + stride_w - 1) // stride_w
                else:
                    w_out = math.floor((current_w - wk_dilation + self.pads[2] + self.pads[3]) / stride_w) + 1
                ho_num = math.floor(tbe_platform.CUBE_MKN[self.weights["dtype"]]["mac"][0] / w_out) + 2
                l1_m = ((ho_num - 1) * stride_h + hk_dilation) * current_w
                max_feature_map_l1 = c0 * l1_m * m_bit_ratio[self.weights["dtype"]]
                if max_feature_map_l1 > l1size_limit_upper:
                    w_right = current_w
                else:
                    w_left = current_w
                current_w = w_left + (w_right - w_left)//2

                if w_left == fmap_w_max:
                    break
            cor_w_range = (fmap_w_min, w_left)
            new_in_range_nchw[W_DIM] = cor_w_range
            to_print = "conv2d fmap ori_range changed from {} to {}.".format(in_range_nchw, new_in_range_nchw)
            warnings.warn(to_print)

        return new_in_range_nchw

    def check_paras(self):
        """
        check original paras
        """
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

        if self.check_unknown_scene(in_shape, out_shape_nchw, w_shape_nchw[N_DIM]):
            in_shape_nchw = [DYNAMIC_FLAG, w_shape_nchw[C_DIM], DYNAMIC_FLAG, DYNAMIC_FLAG]
            in_range_nchw = [(1, None), (w_shape_nchw[C_DIM], w_shape_nchw[C_DIM]), (1, None), (1, None)]
        else:
            self.check_para_dim(in_shape, "in_shape")
            in_shape_nchw, in_range_nchw = self.get_input_nchw(in_shape, self.data_format, in_range)
            if in_shape_nchw[1] == -1:
                in_shape_nchw[1] = w_shape_nchw[1]*self.groups
            self.check_range_valid(in_shape_nchw, in_range_nchw, "fmap", self.data_format)

        cor_in_range_nchw = self.correct_in_range(in_range_nchw, w_shape_nchw)
        self.check_support_valid(in_shape_nchw, w_shape_nchw)
        self.get_attr_nchw(self.data_format)
        y_range, correct_range_flag, new_in_range_nchw = self.get_output_range(w_shape_nchw, cor_in_range_nchw)
        self.check_range_valid(out_shape_nchw, y_range, "output", self.data_format)

        group_para = self.set_group_para(in_shape_nchw, w_shape_nchw, self.dtype)
        in_shape_nchw, w_shape_nchw, in_shape_nc1hwc0, w_shape_frac_z = self._calc_shape(
            in_shape_nchw, w_shape_nchw, new_in_range_nchw, y_range, group_para)
        self.calc_pads(in_shape_nc1hwc0, w_shape_nchw)

        return {"in_shape_nc1hwc0": in_shape_nc1hwc0, "w_shape_frac_z": w_shape_frac_z,
                "w_shape": w_shape_nchw, "group_para": group_para,
                "correct_range_flag": correct_range_flag,
                "new_in_range": new_in_range_nchw}

    def config_paras(self):
        """
        config paras and placeholders
        """

        param = self.check_paras()
        if self.is_tensor == False:
            input_tensor = tvm.placeholder(param.get("in_shape_nc1hwc0"), name="Fmap", dtype=self.dtype)
            weight_tensor = tvm.placeholder(param.get("w_shape_frac_z"), name="Filter", dtype=self.dtype)
            if self.bias:
                bias_tensor = tvm.placeholder((param.get("w_shape")[N_DIM],), name="bias_tensor",
                    dtype=self.bias.get("dtype"))
            else:
                bias_tensor = None
        else:
            input_tensor = self.input_tensor
            weight_tensor = self.weights_tensor
            bias_tensor = self.bias_tensor

        return {"input_tensor": input_tensor, "weight_tensor": weight_tensor, "bias_tensor": bias_tensor,
                "w_shape": param.get("w_shape"), "in_shape_nc1hwc0": param.get("in_shape_nc1hwc0"),
                "w_shape_frac_z": param.get("w_shape_frac_z"), "group_para": param.get("group_para"),
                "correct_range_flag": param.get("correct_range_flag", False), "new_in_range": param.get("new_in_range")}

