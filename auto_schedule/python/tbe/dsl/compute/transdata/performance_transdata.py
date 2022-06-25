#!/usr/bin/env python
# -*- coding: UTF-8 -*-
# Copyright 2021-2021 Huawei Technologies Co., Ltd
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
performance transdata
"""
from tbe import tvm
from tbe.common.utils.shape_util import shape_to_list
from tbe.dsl.base import operation
from tbe.dsl.classifier.transdata.constants import BLOCK, DTYPE_BYTE, DO_TRANSPOSE_PAD
from tbe.dsl.classifier.transdata.constants import BORROW_N_B8B16_BACKWARD, BORROW_N_B8B16_FORWARD
from tbe.dsl.classifier.transdata.constants import BORROW_H_B8B16_BACKWARD, BORROW_H_B8B16_FORWARD
from .transdata_compute import TransdataComputation
from .transdata_op import PadOp, TransposeOp, PadSReshapeOp, SReshapeOp
from .transdata_op import FReshapeOp, SetValueOp, DataMoveOp, DePadOp, PlaceholderOp
from .transdata_op import set_align


class BNBackwardComputation(TransdataComputation):
    """
    BorrowNBackwardComputation
    """

    def __init__(self, tensor, dst_shape, axes_map, pad_value):
        super().__init__(tensor, dst_shape, axes_map, pad_value)
        self._tensor = PlaceholderOp(tensor)
        self._axes_map = dict(sorted(axes_map.items(), key=lambda x: x[1]))
        self.x_idx = 0
        self.x_align_var = BLOCK // DTYPE_BYTE.get(self._tensor.dtype, 1)
        self.c1_idx = \
            list(self._axes_map.keys())[list(self._axes_map.values()).index(self._pad_mode.index(DO_TRANSPOSE_PAD))][0]

    @classmethod
    def get_category(cls):
        """
        Return tag of transdata
        """
        return BORROW_N_B8B16_BACKWARD

    def do_compute(self):
        """
        Main Process
        """
        tensor = self._bn_backward_preprocess()

        tensor = self._bn_backward_process(tensor)

        return self._bn_backward_postprocess(tensor)

    def _bn_backward_preprocess(self):
        # example: (N,C1,H,C0) -> (Nx,C1,H,C0)
        perm, dst_shape = self.calc_align_borrow_axis()
        padded_tensor = PadOp(self._tensor, perm, dst_shape, op_name="pad_n")

        # example: (Nx,C1,H,C0) -> (N1,N0,C1,H,C0)
        perm, dst_shape = self.calc_s_reshape(padded_tensor, self.x_idx, self.x_align_var)
        s_reshaped_tensor = SReshapeOp(padded_tensor, perm, dst_shape, op_name="s_reshape_n")

        # example: (N1,N0,C1,H,C0) -> (N1,C1,H,C0,N0)
        perm, dst_shape = self.calc_transpose_0(s_reshaped_tensor, half=1)
        return TransposeOp(s_reshaped_tensor, perm, dst_shape, op_name="t0")

    def _bn_backward_process(self, tensor):
        # example: (N1,C1,H,C0,N0) -> (N1,H,C1,C0,N0)
        perm, dst_shape = self.calc_transpose_1(tensor, is_forward=False)
        transposed_tensor = TransposeOp(tensor, perm, dst_shape, op_name="t1")

        # example: (N1,H,C1,C0,N0) -> (N1,H,Cx,N0)
        c1 = transposed_tensor.infer_axes(self._tensor, self.c1_idx, half=0)
        perm, dst_shape = self.calc_f_reshape(transposed_tensor, c1)
        f_reshaped_tensor = FReshapeOp(transposed_tensor, perm, dst_shape, op_name="f_reshape_c")

        # example: (N1,H,Cx,N0) -> (N1,H,C,N0)
        tensor = f_reshaped_tensor
        perms, shapes = self.calc_depad(tensor, mode="bn")
        for k, [perm, dst_shape] in enumerate(zip(perms, shapes)):
            tensor = DePadOp(tensor, perm, dst_shape, op_name=f"depad_{k}")
        return tensor

    def _bn_backward_postprocess(self, tensor):
        # example: (N1,H,C,N0) -> (N1,N0,H,C)
        perm, dst_shape = self.calc_transpose_2(tensor, mode="bn")
        transposed_tensor = TransposeOp(tensor, perm, dst_shape, op_name="t2")

        # example:(N1,N0,H,C) -> (Nx,H,C)
        n1 = transposed_tensor.infer_axes(self._tensor, self.x_idx, half=0)
        perm, dst_shape = self.calc_f_reshape(transposed_tensor, n1)
        f_reshaped_tensor = FReshapeOp(transposed_tensor, perm, dst_shape, op_name="f_reshape_n")

        # example: (Nx,H,C) -> (N,H,C)
        perm = list(range(len(f_reshaped_tensor.shape)))
        return DataMoveOp(f_reshaped_tensor, perm, self._dst_shape, "res").tensor


class BNForwardComputation(TransdataComputation):
    """
    BorrowNForwardComputation
    """

    def __init__(self, tensor, dst_shape, axes_map, pad_value):
        super().__init__(tensor, dst_shape, axes_map, pad_value)
        self._tensor = PlaceholderOp(tensor)
        self._axes_map = dict(sorted(axes_map.items()))
        self.x_idx = 0
        self.x_align_var = BLOCK // DTYPE_BYTE.get(self._tensor.dtype, 1)
        self.c_idx = self._pad_mode.index(DO_TRANSPOSE_PAD)

    @classmethod
    def get_category(cls):
        """
        Return tag of transdata
        """
        return BORROW_N_B8B16_FORWARD

    def do_compute(self):
        """
        Main Process
        """
        tensor = self._preprocess()

        tensor = self._process(tensor)

        return self._postprocess(tensor)

    def _preprocess(self):
        # example: (N,H,C) -> (Nx,H,C)
        perm, dst_shape = self.calc_align_borrow_axis()
        padded_tensor = PadOp(self._tensor, perm, dst_shape, op_name="pad_n")

        # example: (N,H,C) -> (N1,N0,H,C)
        perm, dst_shape = self.calc_s_reshape(padded_tensor, self.x_idx, self.x_align_var)
        s_reshaped_tensor = SReshapeOp(padded_tensor, perm, dst_shape, op_name="s_reshape_n")

        # example: (N1,N0,H,C) -> (N1,H,C,N0)
        perm, dst_shape = self.calc_transpose_0(s_reshaped_tensor, half=1)
        return TransposeOp(s_reshaped_tensor, perm, dst_shape, op_name="t0")

    def _process(self, tensor):
        # example: (N1,H,C,N0) -> (N1,Hx,Cx,N0)
        perms, shapes = self.calc_pad(tensor)
        for k, [perm, dst_shape] in enumerate(zip(perms, shapes)):
            tensor = PadOp(tensor, perm, dst_shape, self._pad_value, op_name=f"pad_{k}")

        # example: (N1,Hx,Cx,N0) -> (N1,Hx,C1,C0,N0)
        c = tensor.infer_axes(self._tensor, self.c_idx)
        factor = self._pad_var[c]
        perm, dst_shape = self.calc_s_reshape(tensor, c, factor)
        s_reshaped_tensor = SReshapeOp(tensor, perm, dst_shape, op_name="s_reshape_c")

        # example: (N1,Hx,C1,C0,N0) -> (N1,C1,Hx,C0,N0)
        perm, dst_shape = self.calc_transpose_1(s_reshaped_tensor, is_forward=True)
        return TransposeOp(s_reshaped_tensor, perm, dst_shape, op_name="t1")

    def _postprocess(self, tensor):
        # example: (N1,C1,Hx,C0,N0) -> (N1,N0,C1,Hx,C0)
        perm, dst_shape = self.calc_transpose_2(tensor, mode="bn")
        transposed_tensor = TransposeOp(tensor, perm, dst_shape, op_name="t2")

        # example: (N1,N0,C1,Hx,C0) -> (Nx,C1,Hx,C0)
        n1 = transposed_tensor.infer_axes(self._tensor, self.x_idx, half=0)
        perm, dst_shape = self.calc_f_reshape(transposed_tensor, n1)
        f_reshaped_tensor = FReshapeOp(transposed_tensor, perm, dst_shape, op_name="f_reshape_n")

        # example: (Nx,C1,Hx,C0) -> (N,C1,Hx,C0)
        perm = list(range(len(f_reshaped_tensor.shape)))
        return DataMoveOp(f_reshaped_tensor, perm, self._dst_shape, "res").tensor


class BHForwardComputation(TransdataComputation):
    """
    BorrowHForwardComputation
    """

    def __init__(self, tensor, dst_shape, axes_map, pad_value):
        super().__init__(tensor, dst_shape, axes_map, pad_value)
        self._tensor = PlaceholderOp(tensor)
        self._axes_map = dict(sorted(axes_map.items()))
        self.hi = operation.var_inner("_hi", [1, None])
        self.ho_i = BLOCK // DTYPE_BYTE.get(tensor.dtype, 1)
        self.x_idx = self._borrowed_axes()
        self.c_idx = self._pad_mode.index(DO_TRANSPOSE_PAD)
        self.x_align_var = self.ho_i * self.hi

    @classmethod
    def get_category(cls):
        """
        Return tag of transdata
        """
        return BORROW_H_B8B16_FORWARD

    def do_compute(self):
        """
        Main Process
        """
        tensor = self._bh_preprocess()

        tensor = self._bh_process(tensor)

        return self._bh_postprocess(tensor)

    def _borrowed_axes(self):
        """
        Return index of h base on src-shape
        """
        for k, v in self._axes_map.items():
            if isinstance(v, int) and k != v:
                return k
            if isinstance(v, (list, tuple)) and len(v) == 1 and k != v[0]:
                return k
        return None

    def _calc_pad_s_reshape(self):
        # split h as h1 && h0
        axes = [i if i < self.x_idx else i + 1 for i in range(len(self._axes_map))]
        axes[self.x_idx] = [axes[self.x_idx] - 1, axes[self.x_idx]]
        dst_shape = shape_to_list(self._tensor.shape)
        dst_shape.insert(self.x_idx, tvm.floordiv(set_align(dst_shape[self.x_idx], self.x_align_var), self.hi))
        dst_shape[self.x_idx + 1] = self.hi
        return axes, dst_shape

    def _calc_set_value(self, tensor):
        # pad for borrow-axis
        i = tensor.infer_axes(self._tensor, self.x_idx)
        perm = list(range(len(tensor.shape)))
        perm[i] = [perm[i], ]
        cond = lambda *j: tvm.all(j[i] >= self._src_shape[self.x_idx], j[i] < self._dst_shape[i])
        return perm, tensor.shape, cond

    def _bh_preprocess(self):
        # example: (N, H, C) -> (N, H1, H0, C).
        perm, dst_shape = self._calc_pad_s_reshape()
        padded_tensor = PadSReshapeOp(self._tensor, perm, dst_shape)

        # example: (N, H1, H0, C) -> (N, H0, C, H1)
        perm, dst_shape = self.calc_transpose_0(padded_tensor, half=0)
        return TransposeOp(padded_tensor, perm, dst_shape, op_name="t0")

    def _bh_process(self, tensor):
        # example: (N, H0, C, H1) -> (N, H0, Cx, H1)
        perms, shapes = self.calc_pad(tensor)
        for k, [perm, dst_shape] in enumerate(zip(perms, shapes)):
            tensor = PadOp(tensor, perm, dst_shape, self._pad_value, op_name=f"pad_{k}")

        # example: (N, H0, Cx, H1) -> (N, H0, C1, C0, H1)
        c = tensor.infer_axes(self._tensor, self.c_idx)
        factor = self._pad_var[c]
        perm, dst_shape = self.calc_s_reshape(tensor, c, factor)
        s_reshaped_tensor = SReshapeOp(tensor, perm, dst_shape, op_name="s_reshape_c")

        # example: (N, H0, C1, C0, H1) -> (N, C1, H0, C0, H1)
        perm, dst_shape = self.calc_transpose_1(s_reshaped_tensor, is_forward=True)
        return TransposeOp(s_reshaped_tensor, perm, dst_shape, op_name="t1")

    def _bh_postprocess(self, tensor):
        # example: (N, C1, H0, C0, H1) -> (N, C1, H1, H0, C0)
        perm, dst_shape = self.calc_transpose_2(tensor, mode="bh")
        transposed_tensor = TransposeOp(tensor, perm, dst_shape, op_name="t2")

        # example: (N, C1, H1, H0, C0) -> (N, C1, Hx, C0)
        h1 = transposed_tensor.infer_axes(self._tensor, self.x_idx, half=0)
        perm, dst_shape = self.calc_f_reshape(transposed_tensor, h1)
        f_reshaped_tensor = FReshapeOp(transposed_tensor, perm, dst_shape, op_name="f_reshape_h")

        # example: (N, C1, Hx, C0) -> (N, C1, Hx, C0)
        perm, dst_shape, cond = self._calc_set_value(f_reshaped_tensor)
        padded_tensor = SetValueOp(f_reshaped_tensor, perm, dst_shape, cond, self._pad_value, op_name="pad_h")

        # example: (N, C1, Hx, C0) -> (N, C1, H, C0)
        perm = list(range(len(padded_tensor.shape)))
        return DataMoveOp(padded_tensor, perm, self._dst_shape, "res").tensor


class BHBackwardComputation(TransdataComputation):
    """
    BorrowHBackwardComputation
    """

    def __init__(self, tensor, dst_shape, axes_map, pad_value):
        super().__init__(tensor, dst_shape, axes_map, pad_value)
        self._tensor = PlaceholderOp(tensor)
        self._axes_map = dict(sorted(axes_map.items(), key=lambda x: x[1]))
        self.hi = operation.var_inner("_hi", [1, None])
        self.ho_i = BLOCK // DTYPE_BYTE.get(tensor.dtype, 1)
        self.x_idx = self._borrowed_axes()
        self.x_align_var = self.ho_i * self.hi
        self.c1_idx = \
            list(self._axes_map.keys())[list(self._axes_map.values()).index(self._pad_mode.index(DO_TRANSPOSE_PAD))][0]

    @classmethod
    def get_category(cls):
        """
        Return tag of transdata
        """
        return BORROW_H_B8B16_BACKWARD

    def do_compute(self):
        """
        Main Process
        """
        tensor = self._bh_backward_preprocess()

        tensor = self._bh_backward_process(tensor)

        return self._bh_backward_postprocess(tensor)

    def _borrowed_axes(self):
        """
        Return index of h based on src-shape
        """
        for k, v in self._axes_map.items():
            if isinstance(k, int) and k != v:
                return k
            if isinstance(k, (list, tuple)) and len(k) == 1 and k[0] != v:
                return k[0]
        return None

    def _bh_backward_preprocess(self):
        # example: (N,C1,H,C0) -> (N,C1,Hx,C0)
        perm, dst_shape = self.calc_align_borrow_axis()
        padded_tensor = PadOp(self._tensor, perm, dst_shape, op_name="pad_h")

        # example: (N,C1,Hx,C0) -> (N,C1,H1,H0,C0)
        perm, dst_shape = self.calc_s_reshape(padded_tensor, self.x_idx, self.x_align_var)
        s_reshaped_tensor = SReshapeOp(padded_tensor, perm, dst_shape, op_name="s_reshape_h")

        # example: (N,C1,H1,H0,C0) -> (N,C1,H1,ho_i,hi,C0)
        h0 = s_reshaped_tensor.infer_axes(self._tensor, self.x_idx, half=1)
        perm, dst_shape = self.calc_s_reshape(s_reshaped_tensor, h0, self.hi)
        s_reshaped_tensor = SReshapeOp(s_reshaped_tensor, perm, dst_shape, op_name="s_reshape_h0")

        # example: (N,C1,H1,ho_i,hi,C0) -> (N,C1,H1*ho_i,hi,C0)
        h1 = s_reshaped_tensor.infer_axes(self._tensor, self.x_idx, half=0)
        perm, dst_shape = self.calc_f_reshape(s_reshaped_tensor, h1)
        f_reshaped_tensor = FReshapeOp(s_reshaped_tensor, perm, dst_shape, op_name="f_reshape_h1")

        # example: (N,C1,H1*ho_i,hi,C0) -> (N,C1,hi,C0,H1*ho_i)
        perm, dst_shape = self.calc_transpose_0(f_reshaped_tensor, half=0)
        return TransposeOp(f_reshaped_tensor, perm, dst_shape, op_name="t0")

    def _bh_backward_process(self, tensor):
        # example: (N,C1,hi,C0,H1*ho_i) -> (N,hi,C1,C0,H1*ho_i)
        perm, dst_shape = self.calc_transpose_1(tensor, is_forward=False)
        transposed_tensor = TransposeOp(tensor, perm, dst_shape, op_name="t1")

        # example: (N,hi,C1,C0,H1*ho_i) -> (N,hi,Cx,H1*ho_i)
        c1 = transposed_tensor.infer_axes(self._tensor, self.c1_idx, half=0)
        perm, dst_shape = self.calc_f_reshape(transposed_tensor, c1)
        f_reshaped_tensor = FReshapeOp(transposed_tensor, perm, dst_shape, op_name="f_reshape_c")

        # example: (N,hi,Cx,H1*ho_i) -> (N,hi,C,H1*ho_i)
        tensor = f_reshaped_tensor
        perms, shapes = self.calc_depad(tensor, mode="bh")
        for k, [perm, dst_shape] in enumerate(zip(perms, shapes)):
            tensor = DePadOp(tensor, perm, dst_shape, op_name=f"depad_{k}")
        return tensor

    def _bh_backward_postprocess(self, tensor):
        # example: (N,hi,C,H1*ho_i) -> (N,H1*ho_i,hi,C)
        perm, dst_shape = self.calc_transpose_2(tensor, mode="bh")
        transposed_tensor = TransposeOp(tensor, perm, dst_shape, op_name="t2")

        # example: (N,H1*ho_i,hi,C) -> (N,Hx,C)
        h1 = transposed_tensor.infer_axes(self._tensor, self.x_idx, half=0)
        perm, dst_shape = self.calc_f_reshape(transposed_tensor, h1)
        f_reshaped_tensor = FReshapeOp(transposed_tensor, perm, dst_shape, op_name="f_reshape_h")

        # example: (N,Hx,C) -> (N,H,C)
        perm = list(range(len(f_reshaped_tensor.shape)))
        return DataMoveOp(f_reshaped_tensor, perm, self._dst_shape, "res").tensor
