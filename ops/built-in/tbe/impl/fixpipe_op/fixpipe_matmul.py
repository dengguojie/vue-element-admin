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
from functools import reduce

from impl.fixpipe_op.fixpipe_base import FixpipeBase
from impl.util.platform_adapter import shape_util
from impl.util.platform_adapter import tbe_platform
from impl.util.util_common import ceil
from tbe import tvm


class FixpipeMatmul(FixpipeBase):
    """
    matmul Fixpipe
    """
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
            fixpipe_name += "_out"
            res_reform = tvm.compute(self.output_shape,
                                    lambda *indice: res(*indice),
                                    name=fixpipe_name,
                                    tag=fixpipe_tag)
        return res_reform

    def _check_fc_nd_out(self):
        """
        if out format is NHWC or NC1HWC0, the op type is fc,
        not support NCHW
        """
        out_format = self.output.get("format")
        return out_format in ["NHWC", "NC1HWC0"]

    def _get_post_transform(self):
        """
        get post_transform for op_dict

        Returns
        -------
        string
        """
        if self._is_nz2nd():
            return "NZ2ND"
        return ""

    def _get_output_shape(self):
        """
        get output shape
        """
        shape = self.output.get("shape")
        out_shape = shape
        if self._check_fc_nd_out():
            out_shape = (shape[0], reduce(lambda x, y: x * y, shape[1:]))
        if len(shape) > 5 and self.output.get("format") == "FRACTAL_NZ":
            out_shape = [reduce(lambda x, y: x * y, shape[:-4])] + list(shape[-4:])
        if len(shape) > 3 and self.output.get("format") == "ND":
            out_shape = [reduce(lambda x, y: x * y, shape[:-2])] + list(shape[-2:])
        return out_shape

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
        while self.x1.op.name != "tensor_c_matrix":
            self.x1 = self.x1.op.input_tensors[0]

    def _is_nz2nd(self):
        """
        check nz2nd scene

        Returns
        -------
        bool
        """
        return self.output.get("format") in ("NHWC", "ND", "NC1HWC0")

    def _x2_reform_generate_func(self, x2, input_shape):
        """
        x2 index reform

        Parameters
        ----------
        x2 : tensor
            elewise input
        input_shape : tuple or list
            shape of x1

        Returns
        -------
        lambda description
            new description for elewise input
        """
        if not self._check_fc_nd_out():
            return self._x2_reform_generate_func_default(x2, input_shape)
        # (N,C1,H,W,C0) -> (C1HW,N1,N0,C0)
        x2_n, x2_c1, x2_h, x2_w, x2_c0 = shape_util.shape_to_list(x2.shape)
        x2_l1_shape = (x2_c1 * x2_h * x2_w,
                       ceil(x2_n, tbe_platform.BLOCK_IN),
                       tbe_platform.BLOCK_IN,
                       x2_c0)
        x2_l1 = tvm.compute(
            x2_l1_shape,
            lambda * indice: tvm.select(
                tvm.all(indice[-3] * tbe_platform.BLOCK_IN + indice[-2] < x2_n),
                x2(indice[-3] * tbe_platform.BLOCK_IN + indice[-2],
                   indice[-4] // (x2_h * x2_w),
                   indice[-4] // x2_w % x2_h,
                   indice[-4] % x2_w,
                   indice[-1])
                ),
            name="elewise_l1",
            tag="elewise_l1"
        )
        return self._x2_reform_generate_func_default(x2_l1, input_shape)
