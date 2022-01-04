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
fixpipe process for conv2d fusion
"""
from tbe import tvm
from impl.fixpipe_op.fixpipe_base import FixpipeBase
from impl.fixpipe_op.fixpipe_util import *

NC1MC0_C1_IDX = 1
NC1MC0_C0_IDX = -1
NC1MC0_C0_IDX_REAL = 3
C0_16 = 16
C0_32 = 32
C0_64 = 64


class FixpipeConv2d(FixpipeBase):
    """
    conv2d Fixpipe
    """
    def _get_c0_c1_index(self):
        """
        get c0 c1 index according to format
        """
        return NC1MC0_C0_IDX, NC1MC0_C1_IDX

    def _update_inputs(self):
        """
        skip res_fp32_conv2d tensor
        """
        if self.x1.op.name == "res_fp32_conv2d":
            # skip tensor res_fp32_conv2d, always get tensor res_conv2d as input tensor
            self.x1 = self.x1.op.input_tensors[0]

    def _get_input_dtype(self):
        """
        get fixpipe op input dtype (result type of mad1)
        """
        return self.x1.op.input_tensors[0].dtype

    def _get_output_shape(self):
        """
        get output shape
        """
        shape = self.output.get("shape")
        format = self.output.get("format")
        out_shape = shape
        if len(shape) == 5 and format == "NC1HWC0":
            out_shape = [shape[0], shape[1], shape[2] * shape[3], shape[4]]
        elif len(shape) == 4 and format == "NHWC":
            out_shape = [shape[0], shape[1] * shape[2], shape[3]]
        else:
            raise RuntimeError("error output shape or format")

        return out_shape

    def fixpipe_reform(self, res):
        """
        shape or format transform for fixpipe_op
        """
        if self._is_nz2nd():
            res_reform = tvm.compute(self.output_shape,
                                     lambda n, hw, c: res(n, c // self.input_shape[-1],
                                                          hw, c % self.input_shape[-1]),
                                     name="fixpipe_nz2nd",
                                     tag=FIXPIPE_REFORM_TAG,
                                     attrs=self.attrs)
            return res_reform

        if self._is_channel_merge() or self._is_channel_split():
            name = "fixpipe_channel_merge"
            if self._is_channel_split():
                name = "fixpipe_channel_split"
            res_reform = tvm.compute(self.output_shape,
                                     lambda n, c1, hw, c0:
                                     res(n, (c1 * self.output_shape[-1] + c0) // self.input_shape[-1],
                                         hw,
                                         (c1 * self.output_shape[-1] + c0) % self.input_shape[-1]),
                                     name=name,
                                     tag=FIXPIPE_REFORM_TAG,
                                     attrs=self.attrs)
            return res_reform

        res_reform = tvm.compute(self.output_shape,
                                 lambda *indice: res(*indice),
                                 name="fixpipe_reform_default",
                                 tag=FIXPIPE_REFORM_TAG,
                                 attrs=self.attrs)
        return res_reform

    def _x2_reform_generate_func_anti_quant(self, x2, input_shape):
        """
        x2 index reform with anti_qunt unit
        """
        dim_num = len(input_shape)
        c1_index = NC1MC0_C1_IDX
        c0_index = NC1MC0_C0_IDX_REAL

        x2_shape = shape_to_list(x2.shape)
        x2_c0 = x2_shape[-1]

        if x2_c0 not in [C0_32, C0_64]:
            raise RuntimeError("c0 of x2 should be 32 or 64")

        def lamda_func(*indice):
            new_indice = [0] * dim_num
            for i in range(dim_num):
                if i == c0_index:
                    new_indice[i] = (indice[c1_index] * C0_16 + indice[c0_index]) % x2_c0
                    continue

                if i == c1_index:
                    new_indice[i] = (indice[c1_index] * C0_16 + indice[c0_index]) // x2_c0
                    continue

                new_indice[i] = indice[i]

            return x2(*new_indice)

        return lamda_func

    def _x2_reform_generate_func(self, x2, input_shape):
        """
        x2 index reform
        """
        if self.anti_quant_scale is None:
            return self._x2_reform_generate_func_default(x2, input_shape)

        return self._x2_reform_generate_func_anti_quant(x2, input_shape)
