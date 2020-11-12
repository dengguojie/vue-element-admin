"""
Copyright (C) 2019. Huawei Technologies Co., Ltd. All rights reserved.

This program is free software; you can redistribute it and/or modify
it under the terms of the Apache License Version 2.0.You may not use
this file except in compliance with the License.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
Apache License for more details at
http://www.apache.org/licenses/LICENSE-2.0

lstm_grad
"""

# pylint: disable=locally-disabled,import-error,unused-import,ungrouped-imports
import te.platform as tbe_platform
from te import tik
from te.utils import para_check
from te.utils.error_manager import error_manager_vector
from impl.tik_op_base import TikOpBase

OP_NAME = "GRUV2HiddenGradCell"


def _check_dtype(weight_hidden):
    """
    check parameters dtype
    """
    para_check.check_dtype(weight_hidden["dtype"], ["float16"], "weight_hidden")


def _check_param():
    """
    check parameters
    """
    return True


def _check_attr(gate_order):
    if gate_order not in ['zrh', 'rzh']:
        rule_desc = "gate_order should be zrh or rzh, but current attr is " + gate_order
        error_manager_vector.raise_err_check_params_rules(OP_NAME, rule_desc, 'gate_order', gate_order)


class GRUHiddenGradCell(TikOpBase):
    def __init__(self, tik_instance, dh_pre_t, h, dy, dh, update, reset, new, hidden_new,
                 dh_prev, d_gate_h, dnt_x, t_state, gate_order, kernel_name):
        """ init GRUHiddenGradCell
        """
        super(GRUHiddenGradCell, self).__init__(tik_instance)
        self.t_state = t_state
        self.gate_order = gate_order
        self.kernel_name = kernel_name
        self.device_aicore_num = self.tik_instance.d_profiling.get_aicore_num()
        self.ub_byte_size = tbe_platform.get_soc_spec(tbe_platform.UB_SIZE)
        # [1, output_dim, batch, 16, 16]
        shape = dnt_x["shape"]
        self.fuse_size = self.get_shape_size(shape)
        fuse_shape = (self.fuse_size,)
        self.dtype = h["dtype"]
        self.input_data_size = self.get_data_size(self.dtype)

        # input
        self.dh_pre_t = self.tik_instance.Tensor(self.dtype, fuse_shape, tbe_platform.scope_gm, "dh_pre_t")
        self.h = self.tik_instance.Tensor(self.dtype, fuse_shape, tbe_platform.scope_gm, "h")
        self.dy = self.tik_instance.Tensor(self.dtype, fuse_shape, tbe_platform.scope_gm, "dy")
        self.dh = self.tik_instance.Tensor(self.dtype, fuse_shape, tbe_platform.scope_gm, "dh")
        self.i2 = self.tik_instance.Tensor(self.dtype, fuse_shape, tbe_platform.scope_gm, "update")
        self.r2 = self.tik_instance.Tensor(self.dtype, fuse_shape, tbe_platform.scope_gm, "reset")
        self.n2 = self.tik_instance.Tensor(self.dtype, fuse_shape, tbe_platform.scope_gm, "n2")
        self.n2_mid = self.tik_instance.Tensor(self.dtype, fuse_shape, tbe_platform.scope_gm, "hidden_new")

        # output
        self.dh_prev = self.tik_instance.Tensor(self.dtype, fuse_shape, tbe_platform.scope_gm, "dh_prev")
        d_gate_shape = (3 * self.fuse_size,)
        self.d_gate_h = self.tik_instance.Tensor(self.dtype, d_gate_shape, tbe_platform.scope_gm, "d_gate_h")
        self.dnt_x = self.tik_instance.Tensor(self.dtype, fuse_shape, tbe_platform.scope_gm, "dnt_x")

    def build(self):
        config_map = {"dump_cce_code": False}
        input_list = (self.dh_pre_t, self.h, self.dy, self.dh, self.i2,
                      self.r2, self.n2, self.n2_mid)
        output_list = (self.dh_prev, self.d_gate_h, self.dnt_x)
        self.tik_instance.BuildCCE(self.kernel_name, input_list, output_list, config=config_map)

    def _do_compute(self, input_offset, ele_num):
        """
        t_state: 0 means cur_t == 0 and t_sum = 1 (input )
        t_state: 1 means cur_t == 0 and t_sum > 1
        t_state: 2 means cur_t > 0 and cur_t < t_sum - 1
        t_state: 3 means cur_t == t_sum - 1 and t_sum > 1
        t_state: 4 means cur_t == t  (output dh_prev)
        """
        shape = (ele_num, )
        dh = self.tik_instance.Tensor(self.dtype, shape, tbe_platform.scope_ubuf, "dh")
        self.move_data(dh, self.dh[input_offset], self.dtype, shape)
        if self.t_state > 0:
            dh_pre_t = self.tik_instance.Tensor(self.dtype, shape, tbe_platform.scope_ubuf, "dh_pre_t")
            self.move_data(dh_pre_t, self.dh_pre_t, self.dtype, shape)
            self.vadd_func(dh, dh, dh_pre_t, shape)
        if self.t_state == 4:
            # just cal dh + dh_pre_t in last cell
            return

        dy = self.tik_instance.Tensor(self.dtype, shape, tbe_platform.scope_ubuf, "dy")
        self.move_data(dy, self.dy[input_offset], self.dtype, shape)
        dh_add_dy = dh
        self.vadd_func(dh_add_dy, dh, dy, shape)    # free dy
        i2 = self.tik_instance.Tensor(self.dtype, shape, tbe_platform.scope_ubuf, "i2")
        self.move_data(i2, self.i2[input_offset], self.dtype, shape)

        # cal dh_pre_t for next cell, output to dh_prev
        dh_pre_t = self.tik_instance.Tensor(self.dtype, shape, tbe_platform.scope_ubuf, "dh_pre_t")
        self.vmul_func(dh_pre_t, dh_add_dy, i2, shape)
        self.move_data(self.dh_prev[input_offset], dh_pre_t, self.dtype, shape)

        # cal concat
        one = self.tik_instance.Tensor(self.dtype, shape, tbe_platform.scope_ubuf, "one")
        self.vector_dup_func(one, 1, shape)
        n2 = self.tik_instance.Tensor(self.dtype, shape, tbe_platform.scope_ubuf, "n2")
        self.move_data(n2, self.n2[input_offset], self.dtype, shape)
        power_n2 = self.tik_instance.Tensor(self.dtype, shape, tbe_platform.scope_ubuf, "power_n2")
        self.vmul_func(power_n2, n2, n2, shape)
        one_sub_power_n2 = power_n2
        self.vsub_func(one_sub_power_n2, one, power_n2, shape)
        one_sub_i2 = self.tik_instance.Tensor(self.dtype, shape, tbe_platform.scope_ubuf, "one_sub_i2")
        self.vsub_func(one_sub_i2, one, i2, shape)
        n2_mul_i2 = one_sub_power_n2
        self.vmul_func(n2_mul_i2, one_sub_power_n2, one_sub_i2, shape)
        dn2i = n2_mul_i2
        self.vmul_func(dn2i, n2_mul_i2, dh_add_dy, shape)
        # dn2i -> out
        self.move_data(self.dnt_x[input_offset], dn2i, self.dtype, shape)

        # cal di2
        h1 = self.tik_instance.Tensor(self.dtype, shape, tbe_platform.scope_ubuf, "h1")
        # if self.t_state == 3 or self.t_state == 0:
        #    self.move_data(h1, self.init_h[input_offset], self.dtype, shape)
        # else:
        self.move_data(h1, self.h[input_offset], self.dtype, shape)
        h1_sub_n2 = h1
        self.vsub_func(h1_sub_n2, h1, n2, shape)
        # (1-i2)*i2
        one_i2_mul_i2 = one_sub_i2
        self.vmul_func(one_i2_mul_i2, one_sub_i2, i2, shape)  # free i2
        # (dh2 + dy)*(1-i2)*i2
        dh2_mul_i2 = one_i2_mul_i2
        self.vmul_func(dh2_mul_i2, one_i2_mul_i2, dh_add_dy, shape)
        # (h1-n2)*(dh2 + dy)*(1-i2)*i2
        di2 = dh2_mul_i2
        self.vmul_func(di2, dh2_mul_i2, h1_sub_n2, shape)  # free h1
        # di2 -> out
        if self.gate_order == "zrh":
            offset = input_offset
        else:
            offset = self.fuse_size + input_offset
        self.move_data(self.d_gate_h[offset], di2, self.dtype, shape)

        r2 = self.tik_instance.Tensor(self.dtype, shape, tbe_platform.scope_ubuf, "r2")
        self.move_data(r2, self.r2[input_offset], self.dtype, shape)
        dn2h = dn2i
        self.vmul_func(dn2h, dn2i, r2, shape)
        # dn2h -> out
        self.move_data(self.d_gate_h[self.fuse_size * 2 + input_offset], dn2h, self.dtype, shape)

        one_sub_r2 = r2
        self.vsub_func(one_sub_r2, one, r2, shape)
        n2_mid = self.tik_instance.Tensor(self.dtype, shape, tbe_platform.scope_ubuf, "n2_mid")
        self.move_data(n2_mid, self.n2_mid[input_offset], self.dtype, shape)
        mid_mul_r2 = one_sub_i2
        self.vmul_func(mid_mul_r2, one_sub_r2, n2_mid, shape)
        dr2 = mid_mul_r2
        self.vmul_func(dr2, mid_mul_r2, dn2h, shape)
        # dr2 -> out
        if self.gate_order == "zrh":
            offset = self.fuse_size + input_offset
        else:
            offset = input_offset
        self.move_data(self.d_gate_h[offset], dr2, self.dtype, shape)

    def _get_tiling(self):
        """ get tiling
        """
        ub_max_ele_num = self.ub_byte_size // self.input_data_size
        # align 128 or 64
        align = 256 // self.input_data_size
        if self.fuse_size < align * self.device_aicore_num:
            # fuse align 256
            core_num = self.fuse_size // align
            return {
                "core_num": core_num,
                "loop_num": 1,
                "loop_ele": align,
                "block_size": align,
                "tail_num": 0
            }

        core_num = self.device_aicore_num
        max_block_ele_num = (ub_max_ele_num // 8 // 2 // align) * align
        # fuse align 256
        loop_num = self.fuse_size // (max_block_ele_num * core_num)
        loop_ele = max_block_ele_num
        block_size = loop_ele * loop_num

        tail_num = self.fuse_size - block_size * core_num
        if tail_num == 0:
            return {
                "core_num": core_num,
                "loop_num": loop_num,
                "loop_ele": loop_ele,
                "block_size": block_size,
                "tail_num": tail_num
            }
        tail_loop_ele = (tail_num // core_num + align - 1) // align * align
        tail_core_num = (tail_num + tail_loop_ele - 1) // tail_loop_ele
        tail_last_ele = tail_num % tail_loop_ele if tail_num % tail_loop_ele != 0 else tail_loop_ele
        return {
            "core_num": core_num,
            "loop_num": loop_num,
            "loop_ele": loop_ele,
            "block_size": block_size,
            "tail_num": tail_num,
            "tail_core_num": tail_core_num,
            "tail_loop_ele": tail_loop_ele,
            "tail_last_ele": tail_last_ele
        }

    def compute(self):
        """ do compute
        """
        tiling = self._get_tiling()
        core_num = tiling["core_num"]
        if tiling["loop_num"] > 0:
            with self.tik_instance.for_range(0, core_num, block_num=core_num) as block_idx:
                with self.tik_instance.for_range(0, tiling["loop_num"]) as loop_idx:
                    ele_num = tiling["loop_ele"]
                    base_offset = tiling["block_size"] * block_idx + ele_num * loop_idx
                    self._do_compute(base_offset, ele_num)
        offset = tiling["block_size"] * core_num
        if tiling["tail_num"] > 0:
            core_num = tiling["tail_core_num"]
            with self.tik_instance.for_range(0, core_num, block_num=core_num) as block_idx:
                ele_num = tiling["tail_loop_ele"]
                base_offset = offset + block_idx * ele_num
                with self.tik_instance.if_scope(block_idx < core_num - 1):
                    self._do_compute(base_offset, ele_num)
                with self.tik_instance.else_scope():
                    self._do_compute(base_offset, tiling["tail_last_ele"])


# pylint: disable=locally-disabled,too-many-statements,cell-var-from-loop,unnecessary-lambda
# pylint: too-many-locals,invalid-name,too-many-arguments
def gru_v2_hidden_grad_cell(dh_pre_t, h, dy, dh, update, reset, new, hidden_new,
                            dh_prev, dgate_h, dnt_x, t_state=0, gate_order="zrh",
                            kernel_name="gru_hidden_grad_cell"):
    """
    Calculate the gradient
    Parameters
    -----------
    :param h:
    :param dy:
    :param dh:
    :param update:
    :param reset:
    :param new:
    :param hidden_new:
    :param dh_pre_t: result of (dh2+dy)*i2 at (cur_t -1)
        when t_state > 0, dh = dh + dh_pre_t
    :param dh_prev:
        output real dh_prev when cur_t == t
        otherwise, output dh_pre_t for next cell
    :param dgate_h:
    :param dnt_x:
    :param t_state:
        t_state: 0 means cur_t == 0 and t_sum = 1 (input )
        t_state: 1 means cur_t == 0 and t_sum > 1
        t_state: 2 means cur_t > 0 and cur_t < t_sum - 1
        t_state: 3 means cur_t == t_sum - 1 and t_sum > 1
        t_state: 4 means cur_t == t_sum  (output dh_prev)
    :param gate_order:
    :param kernel_name:
    :return:
    """
    # _check_dtype(weight_hidden)
    # 1. dh dh_pre_t can not be none
    # 2. t_state only has two state(is last cell)
    _check_param()
    _check_attr(gate_order)

    #dh1 = dh_prev
    #dn_i = dnt_x

    tik_instance = tik.Tik(tik.Dprofile())
    cell = GRUHiddenGradCell(tik_instance, dh_pre_t, h, dy, dh, update, reset, new, hidden_new,
                             dh_prev, dgate_h, dnt_x, t_state, gate_order, kernel_name)
    cell.compute()
    cell.build()
