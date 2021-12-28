#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Copyright 2019-2020 Huawei Technologies Co., Ltd
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
auto_schedule template, if user call auto_schedule, this file will choose a
corresponding schedule template for user's compute
"""
import copy
from typing import Any
from typing import Dict
from typing import List

from tbe import tvm
from tbe.common import buildcfg
from tbe.common.platform.platform_info import get_soc_spec
from tbe.common.register import get_op_compute
from tbe.common.context import op_context
from tbe.common.utils.errormgr import get_error_message
from tbe.common.utils import log
from tbe.common.rl_bank import bank_manager
from tbe.common.rl_bank import rl_bank
from tbe.dsl.base import operation
from tbe.dsl.base.var import AttrVarDesc
from tbe.dsl.base.var import Category
from tbe.dsl.base.var import Var
from tbe.dsl.static_schedule.conv_schedule import check_dyn_quantfuse_doubleout
from tbe.dsl.static_schedule.conv_schedule import reget_tensor_list
from tbe.tvm.build_module import build_config
from tbe.tvm.cce_build_module import build_fatbin

from . import CompileInfo
from . import Pattern
from . import pattern_parser
from . import util

CONST = "const"


def is_true(expr, dict_args):
    """
    :param expr: condition
    :param dict_args: error message
    :return: RuntimeError
    """
    if expr:
        raise RuntimeError(dict_args, get_error_message(dict_args))


def schedule_cce(outs, option=None):
    """
    :param outs:
    :param option:
    :return:
    """
    # rl set op res
    bank_manager.set_op_res(outs)

    from tbe.common.buildcfg import get_current_build_config
    original_outs = list(outs) if isinstance(outs, (list, tuple)) else [outs]
    original_outs = reget_tensor_list(original_outs)
    pattern = pattern_parser.get_pattern(outs)
    # set compute pattern to compile info because const tiling need pattern
    operation.add_compile_info_inner(CompileInfo.PATTERN, pattern)
    # set compute pattern
    operation.get_context().get_current_compute().set_pattern(pattern)

    if get_current_build_config("enable_op_prebuild"):
        # prebuild
        op_type = operation.get_context().get_op_type()
        compute = get_op_compute(op_type)

        if compute is None or compute.if_support_fusion() is False:
            fusion_pattern = Pattern.OPAQUE
        else:
            fusion_pattern = pattern

        operation.get_op_context().add_build_res("pattern", fusion_pattern)
        # if CUBE_VECTOR_SPLIT is not empty, the prebuild process goes to the pass side
        if not get_soc_spec("CUBE_VECTOR_SPLIT"):
            return None
    else:
        # try to use rl bank
        try:
            context = operation.get_context()
            if context is not None and context.get_mode() in ("static",):
                ret, rl_sch = rl_bank.query_rl_bank(outs, op_info=None)
                if ret and isinstance(rl_sch, tvm.schedule.Schedule):
                    with operation.schedule() as sch_context:
                        rl_sch.tiling_key = rl_bank.RL_STATIC_TILING_KEY
                        util.add_sch_additional_entry(rl_sch, "context", sch_context)
                    return [rl_sch]
        except Exception as e:
            log.warn("rl bank switch exception: %s, pass!", e)

    tiling_case_func = operation.get_tiling_case(pattern)
    tiling_case_ret = tiling_case_func(original_outs, option)

    schedules = []
    schedule_func = operation.get_schedule(pattern)
    for i, tiling_case in enumerate(tiling_case_ret):
        param_outs = original_outs.copy()
        with operation.schedule() as context:
            context.add("_sch_idx", i)
            if Pattern.CONV2D == pattern:
                sch, real_outs = schedule_func(param_outs, tiling_case)
                ori_outs = real_outs
                param_outs = real_outs
            else:
                sch = schedule_func(param_outs, tiling_case)
                ori_outs = original_outs
            if sch is not None:
                util.add_sch_additional_entry(sch, "original_outs", ori_outs)
                util.add_sch_additional_entry(sch, "real_outs", param_outs)
                util.add_sch_additional_entry(sch, "context", context)
        schedules.append(sch)

    return schedules


# 'pylint: disable=R0914
def build(schedules_list, config_map=None):
    """
    :param schedules_list:
    :param config_map:
    :return:
    """
    from tbe.common.buildcfg import get_current_build_config
    # if CUBE_VECTOR_SPLIT is not empty, the prebuild process goes to the pass side
    if get_current_build_config("enable_op_prebuild") and \
            not get_soc_spec("CUBE_VECTOR_SPLIT"):
        # prebuild
        return

    def get_op_pattern_from_computes():
        """
        get pattern from compute contexts, broadcast and eletwise fix pattern
        use broadcast pattern
        :return:
        """

        # get computes
        op_computes = operation.get_context().get_computes()
        compute_patterns = set()
        for compute in op_computes:
            compute_patterns.add(compute.get_pattern())

        if {Pattern.ELEMWISE, Pattern.BROADCAST} == compute_patterns:
            return Pattern.BROADCAST

        return list(compute_patterns)[-1]

    pattern = get_op_pattern_from_computes()
    # update op pattern
    operation.get_context().set_pattern(pattern)
    operation.add_compile_info_inner(CompileInfo.PATTERN, pattern)

    pointcut_func = operation.get_build_pointcut(pattern)
    if pointcut_func is not None:
        pointcut_func(_build, schedules_list, config_map)
    else:
        _build(schedules_list, config_map)


def _build(schedules_list, config_map):
    Builder(schedules_list, config_map).build()


class Builder:
    """
    class for build
    """
    def __init__(self, schedules_list, config_map):
        self.schedules_list = schedules_list
        self.config_map = config_map

        self.schedules = self._normalize_schedules()
        self.tensors = self._normalize_tensors()

    def build(self):
        self._traverse_context()
        self._traverse_schedules()
        self._call_tvm_build()
        op_context = operation.get_context()
        if op_context.get("_use_cache_tiling") is None or not op_context.get("_use_cache_tiling"): 
            self._handle_compile_info()
        self._handle_addition()

    def _normalize_schedules(self):
        schedules = []
        for sub_schs in self.schedules_list:
            if isinstance(sub_schs, list):
                schedules.extend(sub_schs)
            else:
                schedules.append(sub_schs)

        return schedules

    def _normalize_tensors(self):
        op_context = operation.get_context()
        cpt_contexts = op_context.get_computes()
        tensors = list(self.config_map["tensor_list"])
        if len(cpt_contexts) == 1:
            if not (len(tensors) == 1 and isinstance(tensors[0], (tuple, list))):
                tensors = [tensors]

        is_true(len(cpt_contexts) != len(tensors),
                {"errCode": "E90001",
                "detailed_cause": "The size of compute, build " \
                                   "tensors does not match, they " \
                                   "are [%s] vs [%s]." \
                                   % (len(cpt_contexts), len(tensors))
                })

        return tensors

    def _traverse_context(self):
        op_context = operation.get_context()

        op_vars = op_context.get_vars()
        op_ebv = op_context.get_exclude_bound_vars()
        op_attr_vars_desc = op_context.get_attr_vars_desc()

        # 'ebv' means 'exclude bound vars'
        te_vars_list, ebv_list, attr_vars_desc_list, sch_tensors_list = [], [], [], []
        for i, cpt in enumerate(op_context.get_computes()):
            cpt_vars = cpt.get_vars()
            cpt_ebv = cpt.get_exclude_bound_vars()
            cpt_attr_vars_desc = cpt.get_attr_vars_desc()
            for sch_context in cpt.get_schedules():
                sch_vars = sch_context.get_vars()
                sch_ebv = sch_context.get_exclude_bound_vars()
                sch_attr_vars_desc = sch_context.get_attr_vars_desc()

                te_vars_list.append(op_vars + cpt_vars + sch_vars)
                ebv_list.append(op_ebv + cpt_ebv + sch_ebv)
                attr_vars_desc_list.append(op_attr_vars_desc + cpt_attr_vars_desc + sch_attr_vars_desc)
                sch_tensors_list.append(list(self.tensors[i]))

        lens = [len(self.schedules), len(te_vars_list), len(ebv_list)]
        is_true(len(set(lens)) != 1,
                {"errCode": "E90001",
                "detailed_cause": "The size of schedule, var, and " \
                                  "var_bound does not match, " \
                                  "they are [%s]." % lens
                })

        self.te_vars_list = te_vars_list
        self.ebv_list = ebv_list
        self.attr_vars_desc_list = attr_vars_desc_list
        self.sch_tensors_list = sch_tensors_list

    def _traverse_schedules(self):
        args_list, tiling_keys, valid_schs = [], [], []
        compile_vars, compile_normal_vars, compile_attr_vars, compile_custom_vars = {}, {}, {}, {}

        for sch, te_vars, ebv, attr_vars_desc, sch_tensors in zip(self.schedules,
                                                                  self.te_vars_list,
                                                                  self.ebv_list,
                                                                  self.attr_vars_desc_list,
                                                                  self.sch_tensors_list):
            if sch is None:
                continue

            valid_schs.append(sch)
            real_sch_tensors = self._handle_tensors(sch_tensors, sch)

            var_groups = self._group_vars(te_vars)
            normal_vars = var_groups.get(Category.NORMAL, [])
            attr_vars = var_groups.get(Category.ATTR, [])
            custom_vars = var_groups.get(Category.CUSTOM, [])
            te_vars = normal_vars + attr_vars + custom_vars

            tvm_vars = [x.get_tvm_var() for x in te_vars]
            bounds = [x.get_bound() for x in te_vars]
            for tvm_var, bound in zip(tvm_vars, bounds):
                need_set_range = tvm_var not in ebv and bound is not None
                if need_set_range:
                    sch.set_var_range(tvm_var, *bound)

            args_list.append(real_sch_tensors + tvm_vars)
            tiling_keys.append(sch.tiling_key)

            compile_vars[sch.tiling_key] = te_vars
            compile_normal_vars[sch.tiling_key] = normal_vars
            compile_attr_vars[sch.tiling_key] = attr_vars_desc
            compile_custom_vars[sch.tiling_key] = custom_vars

        self.args_list = args_list
        self.tiling_keys = tiling_keys
        self.valid_schs = valid_schs
        self.compile_vars = compile_vars
        self.compile_normal_vars = compile_normal_vars
        self.compile_attr_vars = compile_attr_vars
        self.compile_custom_vars = compile_custom_vars

    # noinspection PyMethodMayBeStatic
    def _group_vars(self, te_vars):
        # type: (List[Var]) -> Dict[Category, List[Var]]
        ret = {}
        for te_var in te_vars:
            ret.setdefault(te_var.get_category(), []).append(te_var)

        return ret

    # noinspection PyMethodMayBeStatic
    def _handle_tensors(self, sch_tensors, sch):
        real_sch_tensors = sch_tensors.copy()
        original_outs = util.get_sch_additional_entry(sch, "original_outs")
        real_outs = util.get_sch_additional_entry(sch, "real_outs")

        if original_outs != real_outs:
            for tensor_i in original_outs:
                real_sch_tensors.remove(tensor_i)
            for tensor_i in real_outs:
                real_sch_tensors.append(tensor_i)
        real_sch_tensors = check_dyn_quantfuse_doubleout(real_sch_tensors, real_outs)
        return real_sch_tensors

    def _call_tvm_build(self):
        # build config use mode: 1 + m + n
        m_config_items = {}
        if operation.in_dynamic():
            m_config_items.update({"parse_ddr_args": True, "build_fatbin": True})
        if 'build_args' in self.config_map:
            m_config_items.update(self.config_map['build_args'])
        elif 'fusion_build_config' in self.config_map:
            m_config_items.update(self.config_map['fusion_build_config'])
        m_config_items.update(operation.get_build_args())

        dynamic_config = buildcfg.default_buildcfg.dynamic_build_config_dict
        with buildcfg.build_config(**dynamic_config):
            upper_config = buildcfg.get_current_build_config("all")
        build_configs = []

        for sch in self.valid_schs:
            dynamic_single_sch_build_config = copy.deepcopy(upper_config)
            dynamic_single_sch_build_config.update(m_config_items)

            sch_context = util.get_sch_additional_entry(sch, "context")
            n_config_items = sch_context.get("_build_config")
            if n_config_items is not None:
                dynamic_single_sch_build_config.update(n_config_items)

            build_configs.append(build_config(**dynamic_single_sch_build_config))

        if operation.in_dynamic():
            build_fatbin(build_config_list=build_configs,
                         schedule_list=self.valid_schs,
                         args_list=self.args_list,
                         rules=self.tiling_keys,
                         name=self.config_map["name"],
                         target_list="cce")
        else:
            with build_configs[0]:
                tvm.build(self.valid_schs[0],
                          self.args_list[0],
                          target="cce",
                          name=self.config_map["name"]
                          )

    def _handle_compile_info(self):
        def add_vars():
            # key: tiling_key, value: [var_name]
            value = {k: [x.get_name() for x in v] for k, v in self.compile_vars.items()}
            operation.add_compile_info_inner(CompileInfo.VARS, value)

        def add_normal_vars():
            # key: tiling_key, value: [var_name]
            value = {k: [x.get_name() for x in v] for k, v in self.compile_normal_vars.items()}
            operation.add_compile_info_inner(CompileInfo.NORMAL_VARS, value)

        def add_attr_vars():
            def convert(attr_var):
                # type: (AttrVarDesc) -> Dict[str, Any]

                dtype, length = attr_var.dtype, attr_var.length
                if length:
                    dtype = "List" + dtype.capitalize()

                return {
                    "name": attr_var.name,
                    "type": dtype,
                    "length": length or 1
                }

            # key: tiling_key, value: [@see convert_attr_var()]
            value = {k: [convert(x) for x in v] for k, v in self.compile_attr_vars.items()}
            operation.add_compile_info_inner(CompileInfo.ATTR_VARS, value)

        def add_custom_vars():
            # key: tiling_key, value: [var_name]
            value = {k: [x.get_name() for x in v] for k, v in self.compile_custom_vars.items()}
            operation.add_compile_info_inner(CompileInfo.CUSTOM_VARS, value)

        add_vars()
        add_normal_vars()
        add_attr_vars()
        add_custom_vars()

    def _handle_addition(self):
        operation.get_context().add("_tiling_keys", self.tiling_keys)
