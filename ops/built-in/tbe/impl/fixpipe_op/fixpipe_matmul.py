#!/usr/bin/env python
# -*- coding:utf-8 -*-
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
fixpipe fusion with matmul
"""

from tbe import tvm
from impl.fixpipe_op.fixpipe_base import FixpipeBase


class FixpipeMatmul(FixpipeBase):
    """
    matmul Fixpipe
    """
    def _get_c0_c1_index(self):
        """
        get c0 c1 index according to format
        """
        nz_c0_idx = -1
        nz_c1_idx = -4
        return nz_c0_idx, nz_c1_idx

    def _update_inputs(self):
        """
        skip matmul ddr tensor
        """
        if self.x1.op.input_tensors is not None:
            self.x1 = self.x1.op.input_tensors[0]

    def fixpipe_reform(self, res):
        """
        shape or format transform for fixpipe_op
        """
        fixpipe_name = "fixpipe"
        fixpipe_tag = "fixpipe_reform"
        if self._is_nz2nd():
            fixpipe_name += "_nz2nd"
            m_block = self.input_shape[-1]
            n_block = self.input_shape[-2]
            res_reform = tvm.compute(self.output_shape,
                                     lambda *indices: res(*indices[:-2],
                                                          indices[-1] // m_block,
                                                          indices[-2] // n_block,
                                                          indices[-2] % n_block,
                                                          indices[-1] % m_block),
                                     name=fixpipe_name,
                                     tag=fixpipe_tag)
        elif self._is_channel_merge() or self._is_channel_split():
            fixpipe_name += "_channel_merge_split"
            output_n_block = self.output_shape[-1]
            input_n_block = self.input_shape[-1]
            res_reform = tvm.compute(self.output_shape,
                                     lambda *indices:
                                     res(*indices[:-4],
                                         (indices[-4] * output_n_block + indices[-1]) // input_n_block,
                                         indices[-3], indices[-2],
                                         (indices[-4] * output_n_block + indices[-1]) % input_n_block),
                                     name=fixpipe_name,
                                     tag=fixpipe_tag)
        else:
            res_reform = tvm.compute(self.output_shape,
                                    lambda *indice: res(*indice),
                                    name=fixpipe_name,
                                    tag=fixpipe_tag)
        return res_reform
