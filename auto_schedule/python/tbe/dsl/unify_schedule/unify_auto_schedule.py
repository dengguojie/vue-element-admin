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
from typing import Any
from typing import Dict
from typing import List

from tbe.common.utils.errormgr import get_error_message
from tbe.dsl.base import operation
from tbe.dsl.base.var import AttrVarDesc
from tbe.dsl.base.var import Category
from tbe.dsl.base.var import Var
from te import platform as cce
from te import tvm
from te.platform import cce_conf
from te.tvm.build_module import BuildConfigs

from . import CompileInfo
from . import Pattern
from . import pattern_parser
from . import util

CONST = "const"


def schedule_cce(outs, option=None):
    """
    :param outs:
    :param option:
    :return:
    """
    # set "push_status", if "0": main, else: new
    if not operation.get_context().get("push_status"):
        operation.get_context().add("push_status", 0)
    # convert compile_info for tiling.cc, if 0: tiling_data has key, if 1: doesn't has key
    pre_compile_info = operation.get_compile_info()
    if not pre_compile_info:
        operation.add_compile_info("push_status", 0)
    else:
        if "push_status" not in pre_compile_info.keys():
            operation.add_compile_info("push_status", 0)

    original_outs = list(outs) if isinstance(outs, (list, tuple)) else [outs]
    pattern = pattern_parser.get_pattern(outs)
    operation.add_compile_info("_pattern", pattern)
    operation.get_context().set_pattern(pattern)

    if util.get_build_cfg() == "disable":
        # prebuild
        f_m = cce.fusion_manager.fusion_manager
        op_type = operation.get_context().get_op_type()
        compute = operation.get_op_compute(op_type, verbose=True)

        if compute is None or compute.support_fusion is False:
            fusion_pattern = Pattern.OPAQUE
        else:
            fusion_pattern = pattern

        # noinspection PyProtectedMember
        if operation._in_compatible_mode():
            f_m.set_current_op_pattern(fusion_pattern)
        else:
            operation.get_op_context().add_build_res("pattern", fusion_pattern)

        return None

    tiling_case_func = operation.get_tiling_case(pattern)
    tiling_case_ret = tiling_case_func(outs, option)

    schedules = []
    schedule_func = operation.get_schedule(pattern)
    for tiling_case in tiling_case_ret:
        param_outs = original_outs.copy()
        with operation.schedule() as context:
            sch = schedule_func(param_outs, tiling_case)
            if sch is not None:
                util.add_sch_additional_entry(sch, "original_outs", original_outs)
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
    if util.get_build_cfg() == "disable" and \
            not cce_conf.get_soc_spec("CUBE_VECTOR_SPLIT"):
        # prebuild
        return

    pattern = operation.get_context().get_pattern()
    pointcut_func = operation.get_build_pointcut(pattern)
    if pointcut_func is not None:
        pointcut_func(_build, schedules_list, config_map)
    else:
        _build(schedules_list, config_map)


def _build(schedules_list, config_map):
    Builder(schedules_list, config_map).build()


class Builder:
    def __init__(self, schedules_list, config_map):
        self.schedules_list = schedules_list
        self.config_map = config_map

        self.schedules = self._normalize_schedules()
        self.tensors = self._normalize_tensors()

    def build(self):
        self._traverse_context()
        self._traverse_schedules()
        self._call_tvm_build()
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

        if len(cpt_contexts) != len(tensors):
            dict_args = dict()
            dict_args["errCode"] = "E90001"
            dict_args["detailed_cause"] = "The size of compute, build " \
                                          "tensors does not match, they " \
                                          "are [%s] vs [%s]." \
                                          % (len(cpt_contexts), len(tensors))
            raise RuntimeError(dict_args, get_error_message(dict_args))

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
        if len(set(lens)) != 1:
            dict_args = dict()
            dict_args["errCode"] = "E90001"
            dict_args["detailed_cause"] = "The size of schedule, var, and " \
                                          "var_bound does not match, " \
                                          "they are [%s]." % lens
            raise RuntimeError(dict_args, get_error_message(dict_args))

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
        return real_sch_tensors

    def _call_tvm_build(self):
        # build config use mode: 1 + m + n

        m_config_items = {}
        if operation.in_dynamic():
            m_config_items.update({"parse_ddr_args": True, "build_fatbin": True})
        if 'build_args' in self.config_map:
            m_config_items.update(self.config_map['build_args'])
        m_config_items.update(operation.get_build_args())

        cce_build = cce.cce_build
        dynamic_config, static_config = cce_build.dynamic_build_config, cce_build.build_config
        update_func = cce_build.build_config_update_list

        build_configs = [update_func(dynamic_config, m_config_items)]
        for sch in self.valid_schs:
            sch_context = util.get_sch_additional_entry(sch, "context")

            n_config_items = sch_context.get("_build_config")
            if n_config_items is not None:
                m_config_items.update(n_config_items)

            build_config = update_func(dynamic_config, m_config_items)
            build_configs.append(build_config)

        if operation.in_dynamic():
            with BuildConfigs(build_configs):
                tvm.build(self.valid_schs,
                          self.args_list,
                          rules=self.tiling_keys,
                          target="cce",
                          name=self.config_map["name"],
                          )
        else:
            with build_configs[1]:
                tvm.build(self.valid_schs[0],
                          self.args_list[0],
                          target="cce",
                          name=self.config_map["name"],
                          )

    def _handle_compile_info(self):
        def add_vars():
            # key: tiling_key, value: [var_name]
            value = {k: [x.get_name() for x in v] for k, v in self.compile_vars.items()}
            operation.add_compile_info(CompileInfo.VARS, value)

        def add_normal_vars():
            # key: tiling_key, value: [var_name]
            value = {k: [x.get_name() for x in v] for k, v in self.compile_normal_vars.items()}
            operation.add_compile_info(CompileInfo.NORMAL_VARS, value)

        def add_attr_vars():
            def convert(attr_var):
                # type: (AttrVarDesc) -> Dict[str, Any]

                dtype, length = attr_var.dtype, attr_var.length
                if length:
                    dtype = "List" + dtype.capitalize()

                return {
                    "name": attr_var.name,
                    "type": dtype,
                    "length": length or 0
                }

            # key: tiling_key, value: [@see convert_attr_var()]
            value = {k: [convert(x) for x in v] for k, v in self.compile_attr_vars.items()}
            operation.add_compile_info(CompileInfo.ATTR_VARS, value)

        def add_custom_vars():
            # key: tiling_key, value: [var_name]
            value = {k: [x.get_name() for x in v] for k, v in self.compile_custom_vars.items()}
            operation.add_compile_info(CompileInfo.CUSTOM_VARS, value)

        add_vars()
        add_normal_vars()
        add_attr_vars()
        add_custom_vars()

    def _handle_addition(self):
        operation.get_context().add("_tiling_keys", self.tiling_keys)
