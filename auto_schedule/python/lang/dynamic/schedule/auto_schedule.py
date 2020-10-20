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
from te import platform as cce
from te import tvm
from te.platform import operation

from . import pattern_parser
from . import CompileInfo
from . import util

CONST = "const"


def schedule_cce(outs, option=None):
    """
    :param outs:
    :param option:
    :return:
    """
    original_outs = list(outs) if isinstance(outs, (list, tuple)) else [outs]
    pattern = pattern_parser.get_pattern(outs)
    operation.add_compile_info("_pattern", pattern)
    operation.get_context().set_pattern(pattern)

    if util.get_build_cfg() == "disable":
        # prebuild
        f_m = cce.fusion_manager.fusion_manager
        f_m.set_current_op_pattern(pattern)
        return None

    tiling_case_func = operation.get_tiling_case(pattern)
    schedule_func = operation.get_schedule(pattern)
    schedules = []
    for tiling_case in tiling_case_func(outs, option):
        param_outs = original_outs.copy()
        with operation.ScheduleContext():
            sch = schedule_func(param_outs, tiling_case)
            if sch is not None:
                util.add_sch_additional_entry(sch, "original_outs", original_outs)
                util.add_sch_additional_entry(sch, "real_outs", param_outs)
        schedules.append(sch)
    return schedules


# 'pylint: disable=R0914
def build(schedules_list, config_map=None):
    """
    :param schedules_list:
    :param config_map:
    :return:
    """
    if util.get_build_cfg() == "disable":
        # prebuild
        return

    pattern = operation.get_context().get_pattern()
    build_ = operation.get_build(pattern)
    if build_ is not None:
        build_["func"](schedules_list, config_map)
        if build_["break"] is True:
            return

    _build(schedules_list, config_map)


def _build(schedules_list, config_map=None):
    def flatten_sch(_schedules: list):
        for sub_schs in schedules_list:
            if isinstance(sub_schs, list):
                _schedules.extend(sub_schs)
            else:
                _schedules.append(sub_schs)

    schedules = []
    flatten_sch(schedules)

    def check_tensors():
        if len(cpt_contexts) != len(tensors):
            raise RuntimeError(
                "The size of compute, build tensors does not match, "
                "they are {0} vs {1}.".format(len(cpt_contexts), len(tensors)))

    op_context = operation.get_context()
    cpt_contexts = op_context.get_computes()
    tensors = list(config_map["tensor_list"])
    if len(cpt_contexts) == 1:
        if not (len(tensors) == 1 and isinstance(tensors[0], (tuple, list))):
            tensors = [tensors]
    check_tensors()

    te_vars_list = []
    # exclude bound vars list
    ebv_list = []
    op_vars = op_context.get_vars()
    op_ebv = op_context.get_exclude_bound_vars()
    sch_tensors_list = []
    for i, cpt in enumerate(cpt_contexts):
        cpt_vars = cpt.get_vars()
        cpt_ebv = cpt.get_exclude_bound_vars()
        for sch_context in cpt.get_schedules():
            sch_vars = sch_context.get_vars()
            sch_ebv = sch_context.get_exclude_bound_vars()
            te_vars_list.append(op_vars + cpt_vars + sch_vars)
            ebv_list.append(op_ebv + cpt_ebv + sch_ebv)
            sch_tensors_list.append(list(tensors[i]))

    lens = [len(schedules), len(te_vars_list), len(ebv_list)]
    if len(set(lens)) != 1:
        raise RuntimeError(
            "The size of schedule, var, and var_bound does not match, "
            "they are {}.".format(lens))

    def handle_tensors(_sch_tensors, _sch):
        _real_sch_tensors = _sch_tensors.copy()
        original_outs = util.get_sch_additional_entry(_sch, "original_outs")
        real_outs = util.get_sch_additional_entry(_sch, "real_outs")
        if original_outs != real_outs:
            for tensor_i in original_outs:
                _real_sch_tensors.remove(tensor_i)
            for tensor_i in real_outs:
                _real_sch_tensors.append(tensor_i)
        return _real_sch_tensors

    args_list = []
    rules = []
    compile_vars = {}
    valid_schs = []
    for sch, te_vars, ebv, sch_tensors in zip(schedules, te_vars_list, ebv_list, sch_tensors_list):
        if sch is None:
            continue
        valid_schs.append(sch)

        # handle tensors
        real_sch_tensors = handle_tensors(sch_tensors, sch)

        # handle vars
        tvm_vars = [x.get_tvm_var() for x in te_vars]
        bounds = [x.get_bound() for x in te_vars]
        for tvm_var, bound in zip(tvm_vars, bounds):
            need_set_range = tvm_var not in ebv and bound is not None
            if need_set_range:
                sch.set_var_range(tvm_var, *bound)

        args_list.append(real_sch_tensors + tvm_vars)
        rules.append(sch.tiling_key)
        var_names = [x.get_name() for x in te_vars]
        compile_vars[sch.tiling_key] = var_names

    operation.add_compile_info(CompileInfo.VARS, compile_vars)

    build_config_items = {"parse_ddr_args": True,
                          "build_fatbin": True}
    if 'build_args' in config_map:
        build_config_items.update(config_map['build_args'])

    build_config_items.update(operation.get_build_args())

    build_config = cce.cce_build.build_config_update_list(
        cce.cce_build.dynamic_build_config,
        build_config_items)

    with build_config:
        tvm.build(valid_schs, args_list, rules=rules, target="cce",
                  name=config_map["name"])
