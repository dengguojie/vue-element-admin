#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Copyright 2022-2023 Huawei Technologies Co., Ltd
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
conv2d backprop filter schudule util.
"""

from tbe import tvm
from tbe.common.utils.errormgr import error_manager_util
from tbe.dsl.boost_schedule_kit import ScheduleAgent
from tbe.dsl.boost_schedule_kit import ScopeManager


class ScopeManagerReverse(ScopeManager):
    """
    ScopeManager for reverse, __init__ not check leaf_iter_vars and all_iter_vars
    """
    def __init__(self, stage):
        self._stage = stage
        self._axis_unit = {}
        self._active_scopes = []
        self._axis_split_list = []
        self._origin_axis = []
        self._last_attached = None
        self._scope_intrinsic = None
        for axis in stage.leaf_iter_vars:
            if isinstance(axis.dom.extent, tvm.expr.IntImm):
                self._axis_unit[axis] = [1, axis.dom.extent.value]
            else:
                self._axis_unit[axis] = [1, axis.dom.extent]
            self._active_scopes.append(axis)
            self._origin_axis.append(axis)
            self._axis_split_list.append([axis])


class ScheduleAgentReverse(ScheduleAgent):
    """
    ScheduleAgent for reverse __getitem__ use ScopeManagerReverse
    """
    def __getitem__(self, tensor):
        """
        get scope manager of input tensor

        Parameters
        ----------
        tensor : Tensor

        Returns
        -------
        scope_manager

        """
        if isinstance(tensor, tvm.tensor.Tensor):
            key = tensor.op
        else:
            key = tensor
        if self._scope_managers.get(key) is None:
            self._scope_managers[key] = ScopeManagerReverse(self._sch[key])
        return self._scope_managers.get(key)

    def rfactor(self, parent, rfactor_axis):
        """
        Factor a reduction axis in tensor's schedule to an explicit axis.

        Parameters
        ----------
        tensor : Tensor, the tensor to be factored
        axis: IterVar, the reduction axis in the shcedule to be factored

        Returns
        -------
        tensor : Tensor

        """
        scopes = self[parent]
        axis_unit = scopes.get_axis_unit()
        rfactor_res = self._sch.rfactor(parent, rfactor_axis)
        new_axes = self._sch[parent].leaf_iter_vars
        for axis in new_axes:
            if isinstance(axis.dom.extent, tvm.expr.IntImm):
                axis_unit[axis] = [int(axis.dom.extent), int(axis.dom.extent)]
            else:
                axis_unit[axis] = [axis.dom.extent, axis.dom.extent]

        return rfactor_res


class Conv2dbpFilterReverseLoad:
    """
    Reverload load data, reduce the number of copies of MTE1
    Unsupport:
        Scenarios with if conditions, for example:
        1. Non aligned tiling.
        2. fmap_l1 need pad, need reverse fmap_l1 at the same time.
        3. ceil_div(h*w, 16) * 16 % w != 0, need reverse fmap_l1 or grads_l1 at the same time.
    """

    def __init__(self, sch, axis_unit, attach_info, tensor_map, double_buffer_info):
        self._sch = sch
        # the axis in dw_cc and dw_ddr's split factor
        self._axis_unit = axis_unit
        # record which tensors are in each buffer
        self._tensor_map = tensor_map
        # record tensor's attach axis and stage
        self._attach_info = attach_info
        # Record whether the tensor is enabled db
        self._double_buffer_info = double_buffer_info
        # record tensor's leaf_iter_vars
        self._all_axis_dict = {}
        # axis that controls the loading direction
        self._control_reverse_axis = None
        self._reversed_tensor = []
        # record the k-axis that is split due to open db, format is {ori_k_axis: axis_outer, axis_inner}
        self._k_axis_split_dict = {}
        self.splited_axis_dict = {}
        # reverse direction control number
        self.direction_control = 2

    @staticmethod
    def get_axis_split_info(sch_agent, tensor_map):
        """
        get the split info
        params:
            sch_agent: is the class of ScheduleAgent
            tensor_map: Record which tensors are on each buffer
        return:
            split info
        """
        # axis_unit records the size of the axis on dw_ddr and dw_cc
        # This function depends on sch_agent[dw_cc].split and sch_agent[dw_ddr].split
        dw_ddr = tensor_map.get("ddr")
        dw_cc = tensor_map.get("l0c")
        axis_unit = sch_agent[dw_ddr].get_axis_unit()
        axis_unit_dw_cc = sch_agent[dw_cc].get_axis_unit()
        axis_unit.update(axis_unit_dw_cc)

        return axis_unit

    @staticmethod
    def judge_exist_if(pads, l1_height_and_width_info, l1_attach_scope, dw_cc):
        """
        judge exist if condition in IR for al1 and bl1
        params:
            pads: list, all pads value
            l1_height_and_width_info: fmap and grads's tiling_k in l1, and k h
            l1_attach_scope: al1 and bl1 compute_at status
            dw_cc: the tensor in l0c
        return:
            if condition exists in al1 and bl1
        """
        if_status = {
            "al1": False,
            "bl1": False
        }
        l1_tiling_k = l1_height_and_width_info.get("tiling_k")
        l1_hk_size = l1_height_and_width_info.get("size")
        for pad in pads:
            if pad != 0:
                if_status["bl1"] = True
                break
        if l1_attach_scope[0] == dw_cc and l1_tiling_k.get("al1") % l1_hk_size.get("al1")[1] != 0:
            if_status["al1"] = True
        if l1_attach_scope[1] == dw_cc and l1_tiling_k.get("bl1") % l1_hk_size.get("bl1")[1] != 0:
            if_status["bl1"] = True

        return if_status

    @staticmethod
    def _disable_reverse_by_if(grads_reverse, fmap_reverse, if_status):
        """
        if tensor in l1 be reversed and exist if condition in tensor, not support now
        """
        if grads_reverse != [] and grads_reverse[0] != [] and if_status["al1"]:
            return True
        if fmap_reverse != [] and fmap_reverse[0] != [] and if_status["bl1"]:
            return True

        return False

    @staticmethod
    def _error_report(reason):
        dict_args = {}
        dict_args['errCode'] = "E60108"
        dict_args['op_name'] = "conv2d_backprop_filter"
        dict_args['reason'] = reason
        error_manager_util.raise_runtime_error(dict_args)

    def get_control_reverse_axis(self):
        """
        return the axis of control reverse
        """
        return self._control_reverse_axis

    def get_reversed_tensor(self):
        """
        return the tensor be reversed
        """
        return self._reversed_tensor

    def get_split_k_axis_dict(self):
        """
        return the k axis be split by double buffer
        """
        return self._k_axis_split_dict

    def reverse_load(self, l0_at_axis_ddr, if_status):
        """
        the enter of this class
        params:
            l0_at_axis_ddr: dict, l0a and l0b's compute_at axis
                            key is "c_grads_mad_at" and "c_grads_mad_at"
            if_status: dict, If condition exists in IR
                       key is "al1" and "bl1"
        return:
            reverse status, if do reverse success, return True
        """
        # Step1: Get the reverse control axis, the tensor to be reversed, and the tensor to be loaded limit
        control_reverse_axis_closest, reverse_load_flags, limit_load_grads = self._get_reverse_load_flag(l0_at_axis_ddr)
        reverse_load_in_grads, reverse_load_in_fmap = reverse_load_flags
        if self._disable_reverse_by_base_rule(control_reverse_axis_closest):
            return False
        self._control_reverse_axis = control_reverse_axis_closest

        # Step2: Get the axis need be reversed
        grads_reversed_axes = []
        grads_reversed_axis_closest = None
        grads_tensor_reversed = {"l0": self._tensor_map.get("l0a"), "l1": self._tensor_map.get("l1a")}
        if reverse_load_in_grads:
            grads_attach_info = {"l0": self._attach_info.get("l0a"), "l1": self._attach_info.get("l1a")}
            grads_reversed_axes, grads_reversed_axis_closest = self._get_reverse_axis(
                grads_attach_info, control_reverse_axis_closest)

        fmap_reversed_axes = []
        fmap_reversed_axis_closest = None
        fmap_tensor_reversed = {"l0": self._tensor_map.get("l0b"), "l1": self._tensor_map.get("l1b")[0]}
        if reverse_load_in_fmap:
            fmap_attach_info = {"l0": self._attach_info.get("l0b"), "l1": self._attach_info.get("l1b")}
            fmap_reversed_axes, fmap_reversed_axis_closest = self._get_reverse_axis(
                fmap_attach_info, control_reverse_axis_closest)

        if self._disable_reverse_by_if(grads_reversed_axes, fmap_reversed_axes, if_status):
            return False
        # Step3: If open double buffer need split axis by factor 2
        #         l1                        l0
        # [1, 2, 3, 4, 5, 6]  -> [[5, 6], [3, 4], [1, 2]]
        stop_reverse_l0a = self._handle_reverse_with_double_buffer("l0a", reverse_load_in_grads, grads_reversed_axes,
                                                                   grads_reversed_axis_closest)
        stop_reverse_l0b = self._handle_reverse_with_double_buffer("l0b", reverse_load_in_fmap, fmap_reversed_axes,
                                                                   fmap_reversed_axis_closest)
        if stop_reverse_l0a or stop_reverse_l0b:
            return False
        self._after_split_axis_compute_at_again()
        # Step4: Enable reverse
        grads_input_info = [grads_tensor_reversed, control_reverse_axis_closest, grads_reversed_axes]
        fmap_input_info = [fmap_tensor_reversed, control_reverse_axis_closest, fmap_reversed_axes]
        grads_condition, fmap_condition = self._do_reverse_enter(reverse_load_in_grads, reverse_load_in_fmap,
                                                                 grads_input_info, fmap_input_info)
        # Step5: Set limit to reduce the data will load
        if reverse_load_in_grads or reverse_load_in_fmap:
            if limit_load_grads:
                if grads_condition == []:
                    return False
                self._limit_data_read(self._tensor_map.get("l0a"), grads_condition, control_reverse_axis_closest)
            else:
                if fmap_condition == []:
                    return False
                self._limit_data_read(self._tensor_map.get("l0b"), fmap_condition, control_reverse_axis_closest)
        return True

    def _get_reverse_load_flag(self, l0_at_axis_in_ddr):
        """
        Get the reverse control axis, the tensor to be reversed, and the tensor to be loaded limit
        parmas:
            l0_at_axis_in_ddr: dict, l0a and l0b's compute_at axis
                               key is "c_grads_mad_at" and "c_grads_mad_at"
        return:
            control_reverse_axis_closest: Causes repeated loading of the axis closest to the tensor and greater than 1
            enable_reverse_load_flags: reverse_load_in_grads means grads will reverse load
                                       reverse_load_in_fmap means fmap will reverse load
            limit_load_in_grads: if True，means limit grads's data load else fmap
        """
        # get the axis closest to the tensor and greater than 1
        m_effective_axis_in_ddr = self._get_closest_effective_axis(self._tensor_map.get("ddr"),
                                                                   l0_at_axis_in_ddr.get("c_grads_mad_at"))
        n_effective_axis_in_ddr = self._get_closest_effective_axis(self._tensor_map.get("ddr"),
                                                                   l0_at_axis_in_ddr.get("c_fmap_mad_at"))
        m_effective_axis_in_ddr_index = self._get_axis_index_in_tensor(self._tensor_map.get("ddr"),
                                                                       m_effective_axis_in_ddr)
        n_effective_axis_in_ddr_index = self._get_axis_index_in_tensor(self._tensor_map.get("ddr"),
                                                                       n_effective_axis_in_ddr)
        # if l0 at dw_ddr, the axis with the smaller index is the control_reverse_axis, reversed axis is bigger one.
        # if l0 at dw_cc, the axis with the bigger index is the control_reverse_axis, reversed axis is k_axis in dw_cc.
        limit_load_in_grads = m_effective_axis_in_ddr_index > n_effective_axis_in_ddr_index
        limit_load_in_grads = (not limit_load_in_grads) if (self._attach_info.get("l0a").get("stage")
                                                            == self._tensor_map.get("l0c")) else limit_load_in_grads
        reverse_load_in_fmap = False
        reverse_load_in_grads = False
        # limit_load_in_grads is the flag to reduce load data
        if limit_load_in_grads:
            reverse_load_in_grads = True
            control_reverse_axis_closest = n_effective_axis_in_ddr
        else:
            reverse_load_in_fmap = True
            control_reverse_axis_closest = m_effective_axis_in_ddr

        # when tensor compute_at l0c, the k-axes of l0a and l0b are reversed at the same time
        if self._attach_info.get("l0a").get("stage") == self._tensor_map.get("l0c") or self._attach_info.get("l0b").get(
                "stage") == self._tensor_map.get("l0c"):
            reverse_load_in_grads = True
            reverse_load_in_fmap = True

        enable_reverse_load_flags = [reverse_load_in_grads, reverse_load_in_fmap]
        return control_reverse_axis_closest, enable_reverse_load_flags, limit_load_in_grads

    def _disable_reverse_by_base_rule(self, control_reverse_axis_closest):
        """
        disable reverse by double buffer status or control_reverse_axis's value is 1
        """
        disable_reverse_flag = False
        grads_enable_db = self._double_buffer_info.get("l0a")
        fmap_enable_db = self._double_buffer_info.get("l0b")

        # if l0a and l0b in at l0c, need both open double
        if self._tensor_map.get("l0c") in (self._attach_info.get("l0a").get("stage"),
                                           self._attach_info.get("l0b").get("stage")) and (grads_enable_db !=
                                                                                           fmap_enable_db):
            disable_reverse_flag = True
        # todo, and find this when find control_reverse_axis_closest
        if self._axis_unit.get(control_reverse_axis_closest)[-1] == 1:
            disable_reverse_flag = True
        return disable_reverse_flag

    def _get_reverse_axis(self, current_attach_info, control_reverse_axis_closest):
        """
        get all the axis need reversed
        return:
            reversed_axis_outter: reversed axes above l1_tensor
            reversed_axis_inner: reversed axes under tensor l1_tensor
            reversed_axis_closest: reversed axes closest tensor
        """
        l0_attach_stage = current_attach_info.get("l0").get("stage")
        l0_attach_axis = current_attach_info.get("l0").get("axis")
        l1_attach_stage = current_attach_info.get("l1").get("stage")
        l1_attach_axis = current_attach_info.get("l1").get("axis")

        reversed_axis_closest = self._get_closest_effective_axis(l0_attach_stage, l0_attach_axis)
        all_reversed_axis = self._get_relate_axis_between_two_axis(l0_attach_stage, control_reverse_axis_closest,
                                                                   reversed_axis_closest)

        # insert the axis of reduce batch
        # todo, we can do this when get all reversed axes
        reduce_axis_batch = self._sch[self._tensor_map.get("l0c")].op.reduce_axis[0]
        reversed_reduce_axis_batch = self._get_relate_axis(self._tensor_map.get("l0c"), reduce_axis_batch)[:-1]
        if l0_attach_stage == self._tensor_map.get("l0c"):
            all_reversed_axis = self._merge_axis_list_by_index(reversed_reduce_axis_batch, all_reversed_axis,
                                                               l0_attach_stage)
            reversed_axis_closest = all_reversed_axis[-1]

        if l1_attach_stage == self._tensor_map.get("l0c"):
            split_axis = self._get_real_split_axis(l1_attach_axis)
        else:
            split_axis = self._get_closest_effective_axis(l1_attach_stage, l1_attach_axis)

        if split_axis in all_reversed_axis:
            split_index = all_reversed_axis.index(split_axis) + 1
            reversed_axis_outter = all_reversed_axis[:split_index]
            reversed_axis_inner = all_reversed_axis[split_index:]
        else:
            reversed_axis_outter = []
            reversed_axis_inner = all_reversed_axis

        reversed_axis_outter = self._filter_axis(reversed_axis_outter)
        reversed_axis_inner = self._filter_axis(reversed_axis_inner)
        return [reversed_axis_outter, reversed_axis_inner], reversed_axis_closest

    def _handle_reverse_with_double_buffer(self, buffer_name, reverse_load_flag, reversed_axes, reversed_axis_closest):
        """
        If open double buffer need split axis by factor 2
        params:
            buffer_name: value is "l0a" or "l0b"
            reverse_load_flag: the flag enable reverse load
            reversed_axes: all axes will be reversed, may renew after split some axis
            reversed_axis_closest: the axis closest to the tensor and greater than 1
        return:
            the flag disable double, if true means disable reverse
        """
        # the number 2 means enable double buffer
        enable_double_buffer = (self._double_buffer_info.get(buffer_name) == 2)
        if enable_double_buffer and reverse_load_flag and reversed_axes != [] and (reversed_axes[-1] != []) and (
                reversed_axis_closest is not None):
            # the number 2 means axis value is 2
            # todo, this filter condition can try to delete
            if self._axis_unit.get(reversed_axis_closest)[-1] % self.direction_control != 0 or self._axis_unit.get(
                    reversed_axis_closest)[-1] == 2:
                return True

            # if reversed_axis_closest not split need do split
            if reversed_axis_closest not in self.splited_axis_dict:
                split_factor = 2
                l0_attach_stage = self._attach_info.get(buffer_name).get("stage")
                reversed_outer, not_reversed_inner = self._sch[l0_attach_stage].split(reversed_axis_closest,
                                                                                      split_factor)
                new_extern = self._axis_unit.get(reversed_axis_closest)[-1] // split_factor
                self._axis_unit[reversed_outer] = [new_extern, new_extern]
                self._axis_unit[not_reversed_inner] = [split_factor, split_factor]
                self.splited_axis_dict[reversed_axis_closest] = [l0_attach_stage, reversed_outer, not_reversed_inner]
                # record the k-axis be splited
                if l0_attach_stage == self._tensor_map.get("l0c"):
                    self._k_axis_split_dict[reversed_axis_closest] = [reversed_outer, not_reversed_inner]
            else:
                _, reversed_outer, _ = self.splited_axis_dict.get(reversed_axis_closest)

            # renew the axis need reversed
            if reversed_axis_closest in reversed_axes[-1]:
                ori_axis_index = reversed_axes[-1].index(reversed_axis_closest)
                reversed_axes[-1][ori_axis_index] = reversed_outer
        return False

    def _after_split_axis_compute_at_again(self):
        """
        The original axis is splited and needs to be re-compute_at into the inner axis
        """
        for stage_name, single_attach_info in self._attach_info.items():
            attach_axis = single_attach_info.get("axis")
            if attach_axis in self.splited_axis_dict:
                cur_stage = self.splited_axis_dict.get(attach_axis)[0]
                cur_axis = self.splited_axis_dict.get(attach_axis)[-1]
                re_compute_at_tensors = self._tensor_map.get(stage_name)
                if not isinstance(re_compute_at_tensors, list):
                    tensors = [re_compute_at_tensors, ]
                else:
                    tensors = re_compute_at_tensors
                for tensor in tensors:
                    self._sch[tensor].compute_at(self._sch[cur_stage], cur_axis)

    def _do_reverse_enter(self, reverse_load_in_grads, reverse_load_in_fmap, grads_input_info, fmap_input_info):
        grads_load_condition = []
        fmap_load_condition = []
        if reverse_load_in_grads:
            grads_load_condition = self._do_reverse(*grads_input_info, self._attach_info.get("l0a").get("stage"))
        if reverse_load_in_fmap:
            fmap_load_condition = self._do_reverse(*fmap_input_info, self._attach_info.get("l0b").get("stage"))

        return grads_load_condition, fmap_load_condition

    def _do_reverse(self, tensor_reversed, control_reverse_axis_closest, reversed_axes, l0_attach_stage):
        reversed_axis_outter = reversed_axes[0]
        reversed_axis_inner = reversed_axes[1]
        if reversed_axis_inner == []:
            return []
        l0_tensor_reversed = tensor_reversed.get("l0")
        l1_tensor_reversed = tensor_reversed.get("l1")
        condition_axis = reversed_axis_outter + reversed_axis_inner
        for reversed_axis in condition_axis:
            self._sch[l0_tensor_reversed].reverse(reversed_axis,
                                                  control_reverse_axis_closest % self.direction_control == 0)
            self._record_reversed_tensor(l0_tensor_reversed)
            # The order in which the n or m directions are loaded into the mad unit is reversed,
            # so the output is also reversed
            if l0_attach_stage == self._tensor_map.get("ddr"):
                self._sch[self._tensor_map.get("ddr")].reverse(
                    reversed_axis, control_reverse_axis_closest % self.direction_control == 0)
                self._record_reversed_tensor(self._tensor_map.get("ddr"))
            elif l0_attach_stage == self._tensor_map.get("l0c"):
                self._sch[self._tensor_map.get("l0c")].reverse(
                    reversed_axis, control_reverse_axis_closest % self.direction_control == 0)
                self._record_reversed_tensor(self._tensor_map.get("l0c"))

        for reversed_axis in reversed_axis_outter:
            self._sch[l1_tensor_reversed].reverse(reversed_axis,
                                                  control_reverse_axis_closest % self.direction_control == 0)
            self._record_reversed_tensor(l1_tensor_reversed)

        return condition_axis

    def _do_data_load_limit_enter(self, ):
        pass

    def _record_reversed_tensor(self, tensor):
        if tensor not in self._reversed_tensor:
            self._reversed_tensor.append(tensor)

    def _limit_data_read(self, tensor_reversed, load_condition_axes, control_reverse_axis):
        condition_need_load_data = self._gen_limit_load_condition(load_condition_axes, control_reverse_axis)

        self._sch[tensor_reversed].set_store_predicate(condition_need_load_data)
        self._sch[tensor_reversed].mem_unique()

    def _gen_limit_load_condition(self, load_condition_axes, control_reverse_axis):
        condition_skip_first_data_normal_list = []
        condition_skip_first_data_reverse_list = []
        condition_init_top_data_normal_list = [control_reverse_axis.var == 0]
        condition_init_top_data_reverse_list = [control_reverse_axis.var == 0]
        for axis in load_condition_axes:
            cur_extend = self._axis_unit.get(axis)[-1] - 1
            condition_skip_first_data_normal_list.append(axis.var > 0)
            condition_skip_first_data_reverse_list.append(axis.var < cur_extend)

            condition_init_top_data_reverse_list.append(axis.var == cur_extend)
            condition_init_top_data_normal_list.append(axis.var == 0)

        condition_init_top_data_normal = tvm.all(*condition_init_top_data_normal_list)
        condition_init_top_data_reverse = tvm.all(*condition_init_top_data_reverse_list)

        condition_skip_first_data_normal = tvm.any(*condition_skip_first_data_normal_list)
        condition_skip_first_data_reverse = tvm.any(*condition_skip_first_data_reverse_list)
        condition_load_data_normal = tvm.any(condition_init_top_data_normal, condition_skip_first_data_normal)
        condition_load_data_reverse = tvm.any(condition_init_top_data_reverse, condition_skip_first_data_reverse)
        condition_need_load_data = tvm.select(control_reverse_axis % self.direction_control == 0,
                                              condition_load_data_reverse, condition_load_data_normal)

        return condition_need_load_data

    def _get_relate_axis(self, tensor, axis_key):
        """
        get the axis whose name contain axis_key
        """
        ori_axis_key = axis_key.var.name.split(".")[0]
        axis_list = []
        for axis in self._sch[tensor].leaf_iter_vars:
            if axis.var.name.find("{}{}".format(ori_axis_key, '.')) == 0:
                axis_list.append(axis)
        return axis_list

    def _get_closest_effective_axis(self, base_tensor, base_axis):
        effective_axis = base_axis
        relate_axes = self._get_relate_axis(base_tensor, base_axis)
        if base_axis in relate_axes:
            index_end = relate_axes.index(base_axis)
        else:
            reason = "the axis of {} is not recorded".format(base_axis.var.name)
            self._error_report(reason)

        for relate_axis in reversed(relate_axes[:index_end + 1]):
            current_extend = self._axis_unit.get(relate_axis)[-1]
            if current_extend is None:
                reason = "split info not find, current axis is {}".format(relate_axis.var.name)
                self._error_report(reason)
            if current_extend > 1:
                effective_axis = relate_axis
                break
        return effective_axis

    def _get_axis_index_in_tensor(self, base_tensor, base_axis):
        if base_tensor not in self._all_axis_dict:
            all_axis = list(self._sch[base_tensor].leaf_iter_vars)
            self._all_axis_dict[base_tensor] = all_axis
        else:
            all_axis = self._all_axis_dict.get(base_tensor)

        if base_axis not in all_axis:
            reasion = "the axis of {} is not find in tensor {}".format(base_axis.var.name, base_tensor.op.name)
            self._error_report(reasion)
        return all_axis.index(base_axis)

    def _get_relate_axis_between_two_axis(self, base_tensor, start_axis, end_axis):
        all_axis = list(self._sch[base_tensor].leaf_iter_vars)
        if start_axis not in all_axis:
            start_index = 0
        else:
            start_index = all_axis.index(start_axis)
        end_index = all_axis.index(end_axis) + 1

        ori_axis_key = end_axis.var.name.split(".")[0]
        axis_list = []
        for axis in all_axis[start_index:end_index]:
            if axis.var.name.find("{}{}".format(ori_axis_key, ".")) == 0:
                axis_list.append(axis)
        return axis_list

    def _filter_axis(self, axis_list):
        """
        filter the axis in axis_list
        """
        result = []
        for axis in axis_list:
            extend_value = self._axis_unit.get(axis)[-1]
            if extend_value is not None and extend_value <= 1:
                continue
            if ".fused" in axis.var.name:
                continue
            result.append(axis)
        return result

    def _merge_axis_list_by_index(self, axis_inserted, list_inserted, attach_stage):
        index_axis_info = []
        for axis in axis_inserted:
            index_axis_info.append((self._get_axis_index_in_tensor(attach_stage, axis), axis))
        for axis in list_inserted:
            index_axis_info.append((self._get_axis_index_in_tensor(attach_stage, axis), axis))
        insert_by_order = sorted(index_axis_info, key=lambda d: d[0])
        return list(zip(*insert_by_order))[-1]

    def _get_real_split_axis(self, current_axis):
        tensor_l0c = self._tensor_map.get("l0c")
        all_reversed_axis = list(self._sch[tensor_l0c].leaf_iter_vars)
        result = current_axis
        index = all_reversed_axis.index(current_axis)
        all_reversed_axis = all_reversed_axis[:index+1]
        for axis in all_reversed_axis[::-1]:
            if self._axis_unit.get(axis)[-1] != 1:
                result = axis
                break
        return result
