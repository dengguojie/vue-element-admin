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
fixpipe base functions
"""
from typing import List
from tbe import tvm
from tbe.tvm.tensor import Tensor
from tbe.common.utils import log
from tbe.common.utils import shape_to_list
from impl.fixpipe_op.fixpipe_util import *


class FixpipeBase(object):
    """
    fixpipe base class
    """
    def __init__(self, op_type: str, x1: Tensor, x2: (Tensor, None), quant_scale_0: (Tensor, None),
                 relu_weight_0: (Tensor, None), clip_value_0: (Tensor, None),
                 quant_scale_1: (Tensor, None), relu_weight_1: (Tensor, None),
                 clip_value_1: (Tensor, None),
                 anti_quant_scale: (Tensor, None), anti_quant_offset: (Tensor, None),
                 output: dict, fusion_op_list: List[str], unit_list: List[str], eltwise_mode: str):
        """
        FixpipeBase init func
        """
        # set op input params
        self.op_type = op_type
        self.x1 = x1
        self.x2 = x2
        self.quant_scale_0 = quant_scale_0
        self.relu_weight_0 = relu_weight_0
        self.clip_value_0 = clip_value_0
        self.quant_scale_1 = quant_scale_1
        self.relu_weight_1 = relu_weight_1
        self.clip_value_1 = clip_value_1
        self.anti_quant_scale = anti_quant_scale
        self.anti_quant_offset = anti_quant_offset
        self.output = output
        self.fusion_op_list = fusion_op_list
        self.unit_list = unit_list
        self.eltwise_mode = eltwise_mode

        # set vector tendor flag
        self.quant_scale_0_vector_flag = is_vector_input(self.quant_scale_0)
        self.relu_weight_0_vector_flag = is_vector_input(self.relu_weight_0)
        self.quant_scale_1_vector_flag = is_vector_input(self.quant_scale_1)
        self.relu_weight_1_vector_flag = is_vector_input(self.relu_weight_1)

        self.vector_inputs_dict = {
            QUANT_SCALE_0_STR: self.quant_scale_0,
            QUANT_SCALE_1_STR: self.quant_scale_1,
            RELU_WEIGHT_0_STR: self.relu_weight_0,
            RELU_WEIGHT_1_STR: self.relu_weight_1,
            ELTWISE_SRC_STR: self.x2
        }

        self.input_dtype = ""
        self.input_shape = []
        self.output_dtype = ""
        self.output_shape = []
        self.attrs = {}
        self.op_dict = {}

    def _get_params(self):
        """
        get necessary info for fixpipe_op
        """
        self.input_dtype = self._get_input_dtype()
        self.input_shape = self._get_input_shape()
        self.output_dtype = self._get_output_dtype()
        self.output_shape = self._get_output_shape()
        self.attrs = self._get_attrs()
        self.op_dict = self._get_op_dict()

    def _get_output_shape(self):
        """
        get output shape
        """
        return self.output.get("shape")

    def _get_c0_c1_index(self):
        """
        get c0 c1 index
        """
        return NC1HWC0_C0_IDX, NC1HWC0_C1_IDX

    def _get_input_shape(self):
        """
        get input tensor shape
        """
        return shape_to_list(self.x1.shape)

    def _get_output_dtype(self):
        """
        get output dtype
        """
        return self.output.get("dtype")

    def _get_input_dtype(self):
        return self.x1.dtype

    def _get_pre_conv(self):
        """
        get pre_conv for op_dict
        """
        def _get_pre_dst_dtype(quant_scale_1, output_dtype):
            dst_pre_conv_dtype_ = output_dtype
            if quant_scale_1 is not None:
                dst_pre_conv_dtype_ = DTYPE_FLOAT16
            return dst_pre_conv_dtype_

        conv_mode = ""
        if is_vector_input(self.quant_scale_0):
            conv_mode += "V"

        dst_pre_conv_dtype = _get_pre_dst_dtype(self.quant_scale_1, self.output_dtype)
        conv_mode += DTYPE_TRANS_MAP.get(self.input_dtype) + "2" + DTYPE_TRANS_MAP.get(
            dst_pre_conv_dtype)

        if conv_mode in PASS_PRE_CONVERT_MODE:
            return ""

        if conv_mode not in PRE_CONVERT_MODE:
            raise RuntimeError("{} is not supported for fixpipe pre_conv".format(conv_mode))

        return conv_mode

    def _get_pre_activation(self):
        """
        get pre_activation for op_dict
        """
        if self.relu_weight_0 is not None:
            if is_scaler_input(self.relu_weight_0):
                return SCALAR_RELU_MODE
            return VECTOR_RELU_MODE

        if PRE_ACT_UNIT_STR in self.unit_list:
            return NORMAL_RELU_MODE

        return ""

    def _get_post_anti_quant(self):
        """
        get post_anti_quant for op_dict
        """
        if self.anti_quant_scale is None:
            return ""

        anti_quant_dtype = self.x2.dtype
        if anti_quant_dtype not in ANTI_QUANT_MAP.keys():
            raise RuntimeError("{} is not supported for fixpipe anti_quant".format(anti_quant_dtype))

        return ANTI_QUANT_MAP.get(anti_quant_dtype)

    def _get_post_eltwise(self):
        """
        get post_eltwise for op_dict
        """
        if self.x2 is None:
            if self.eltwise_mode != "":
                raise RuntimeError("eltwise_mode should be SUB or ADD when x1 is not None")

            return ""

        return self.eltwise_mode

    def _get_post_activation(self):
        """
        get post_activation for op_dict
        """
        if self.relu_weight_1 is not None:
            if is_scaler_input(self.relu_weight_1):
                return SCALAR_RELU_MODE
            return VECTOR_RELU_MODE

        if POST_ACT_UNIT_STR in self.unit_list:
            return NORMAL_RELU_MODE

        return ""

    def _get_post_quant(self):
        """
        get post_quant for op_dict
        """
        if self.quant_scale_1 is None:
            return ""

        conv_mode = ""
        if is_vector_input(self.quant_scale_1):
            conv_mode += "V"

        conv_mode += DTYPE_TRANS_MAP.get(DTYPE_FLOAT16) + "2" + DTYPE_TRANS_MAP.get(self.output_dtype)
        if conv_mode not in POST_QUANT_MODE:
            raise RuntimeError("{} is not supported for fixpipe post_quant".format(conv_mode))

        return conv_mode

    def _get_post_transform(self):
        """
        get post_transform for op_dict
        """
        if self.output.get("format") in ("NHWC", "ND"):
            return "NZ2ND"
        return ""

    def _is_nz2nd(self):
        """
        check nz2nd scene
        """
        if self.output.get("format") in ("NHWC", "ND"):
            return True
        return False

    def _is_channel_merge(self):
        """
        check channel merge scene
        """
        if self._is_nz2nd():
            return False

        if self.output_dtype in ["int8", "int4"]:
            return True
        return False

    def _is_channel_split(self):
        """
        check channel spilt scene
        """
        if self._is_nz2nd():
            return False

        if self.output_dtype == DTYPE_FLOAT32:
            return True
        return False

    def _get_vector_tensors(self):
        """
        get vector tensors from inputs
        """
        vector_params = []
        vector_tensors = []

        for input_name in self.vector_inputs_dict.keys():
            input_tensor = self.vector_inputs_dict.get(input_name)
            if is_vector_input(input_tensor):
                vector_params.append(input_name)
                vector_tensors.append(input_tensor)

        return vector_params, vector_tensors

    def _get_op_dict(self):
        """
        get op_dict for tvm.fixpipe_op
        """
        op_dict = {
            "pre_conv": self._get_pre_conv(),
            "pre_activation": self._get_pre_activation(),
            "post_anti_quant": self._get_post_anti_quant(),
            "post_eltwise": self._get_post_eltwise(),
            "post_activation": self._get_post_activation(),
            "post_quant": self._get_post_quant(),
            "post_transform": self._get_post_transform()
        }
        log.debug("fixpipe op_dict:{}".format(op_dict))
        return op_dict

    def _get_attrs(self):
        """
        get attrs for fixpipe
        """
        attrs = {}
        vector_params, vector_tensors = self._get_vector_tensors()
        attrs["vector_params"] = vector_params
        attrs["vector_tensors"] = vector_tensors
        attrs["nz2nd_flag"] = self._is_nz2nd()
        log.debug("fixpipe attrs:{}".format(attrs))
        return attrs

    def _update_inputs(self):
        """
        update op input for special scenes
        """
        pass

    def _param_check(self):
        """
        check op input params
        """
        pass

    def fixpipe_op_compute(self):
        """
        main fixpipe compute
        default input format is NC1HWC0
        """
        c0_index, c1_index = self._get_c0_c1_index()
        max_index = len(self.input_shape) - 1
        if c0_index > max_index or c1_index > max_index:
            raise RuntimeError("c0_index or c1_index is out of range")

        fixpipe_op = tvm.compute(self.input_shape,
                                 lambda *indices:
                                 tvm.fixpipe_op(self.x1(*indices),
                                                self.output_dtype,
                                                pre_conv_param=self.quant_scale_0(0, indices[c1_index], 0, 0, indices[c0_index]) if self.quant_scale_0_vector_flag else get_input_scalar_value(self.quant_scale_0),
                                                pre_relu_param=self.relu_weight_0(0, indices[c1_index], 0, 0, indices[c0_index]) if self.relu_weight_0_vector_flag else get_input_scalar_value(self.relu_weight_0),
                                                pre_clip_relu_param=get_input_scalar_value(self.clip_value_0),
                                                post_eltwise_src=self.x2(*indices) if self.x2 is not None else self.x2,
                                                post_anti_quant_scale=get_input_scalar_value(self.anti_quant_scale),
                                                post_anti_quant_offset=get_input_scalar_value(self.anti_quant_offset),
                                                post_clip_relu_param=get_input_scalar_value(self.clip_value_1),
                                                post_quant_param=self.quant_scale_1(0, indices[c1_index], 0, 0, indices[c0_index]) if self.quant_scale_1_vector_flag else get_input_scalar_value(self.quant_scale_1),
                                                post_relu_param=self.relu_weight_1(0, indices[c1_index], 0, 0, indices[c0_index]) if self.relu_weight_1_vector_flag else get_input_scalar_value(self.relu_weight_1),
                                                op_dict=self.op_dict),
                                 name=FIXPIPE_OP_TAG,
                                 tag=FIXPIPE_OP_TAG,
                                 attrs=self.attrs)
        return fixpipe_op

    def fixpipe_reform(self, res):
        """
        shape or format transform for fixpipe_op
        """
        res_reform = tvm.compute(self.output_shape,
                                 lambda *indice: res(*indice),
                                 name=FIXPIPE_REFORM_TAG,
                                 tag=FIXPIPE_REFORM_TAG)
        return res_reform

    def fixpipe_compute(self):
        """
        fixpipe compute
        """
        self._update_inputs()
        self._get_params()
        self._param_check()

        fixpipe_op = self.fixpipe_op_compute()
        fixpipe_reform = self.fixpipe_reform(fixpipe_op)

        return fixpipe_reform
