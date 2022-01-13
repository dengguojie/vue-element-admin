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
fixpipe fusion for conv2d_bp_filter
"""

from tbe import tvm
from impl.fixpipe_op.fixpipe_base import FixpipeBase


class FixpipeConv2dBackpropFilter(FixpipeBase):
    """
    conv2d_backprop_filter Fixpipe
    """
    @staticmethod
    def _get_c0_c1_index():
        """
        get c0 c1 index according to format
        """
        return -1, 1

    def _update_inputs(self):
        """
        skip channel_split tensor and get dw_cc as input tensor.
        """
        if self.x1.op.name == "dw_c_split":
            self.x1 = self.x1.op.input_tensors[0]

    def _get_output_shape(self):
        """
        get output shape
        """
        shape = self.output.get("shape")
        out_shape = shape
        if len(shape) == 4 and self.output.get("format") == "NHWC":
            out_shape = [shape[0], shape[1] * shape[2], shape[3]]
        elif len(shape) == 4 and self.output.get("format") == "FRACTAL_Z":
            # (C1HW, N1, N0, C0) -> (real_g, fkk, Cout_g, fmap_c0)
            out_shape = [1, shape[0], shape[1] * shape[2], shape[3]]
        else:
            raise RuntimeError("error output shape or format")

        return out_shape

    def fixpipe_reform(self, res):
        """
        shape or format transform for fixpipe_op
        """
        fixpipe_op_tag = "fixpipe"
        fixpipe_reform_tag = "fixpipe_reform"
        fixpipe_no_trans_tag = "fixpipe_no_trans"
        self.attrs["kernel_name"] = self.x1.op.attrs["kernel_name"]
        if self._is_nz2nd():
            _, _, cout_g, fmap_c0 = tuple(i.value for i in self.x1.shape)
            _, hk_wk, _ = self.output_shape
            res_reform = tvm.compute(self.output_shape,
                                     lambda n_idx, hw_idx, c_idx:
                                        res(n_idx // cout_g,
                                            c_idx // fmap_c0 * hk_wk + hw_idx,
                                            n_idx % cout_g,
                                            c_idx % fmap_c0),
                                     name=fixpipe_op_tag + "_nz2nd",
                                     tag=fixpipe_reform_tag,
                                     attrs=self.attrs)

            return res_reform

        res_reform = tvm.compute(self.output_shape,
                                 lambda *indice: res(*indice),
                                 name=fixpipe_op_tag + "_out",
                                 tag=fixpipe_no_trans_tag,
                                 attrs=self.attrs)
        return res_reform
