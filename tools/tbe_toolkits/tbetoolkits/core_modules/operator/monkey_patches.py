#!/usr/bin/env python3
"""
Sometimes monkey patch is the only way to achieve your objective
"""
# Standard Packages
import logging
import sys
import copy
from types import ModuleType
from typing import NoReturn
from typing import Callable
from typing import Sequence


class StubBuildConfig:
    """
    Fake BuildConfig
    """

    def __init__(self, parent_func: Callable, cfg_maps: dict):
        self.__parent = parent_func
        self.__my_cfg_maps = cfg_maps

    def __dir__(self):
        # noinspection PyUnresolvedReferences
        return dir(self.__parent())

    def get_attr(self, item):
        if item in self.__my_cfg_maps:
            return self.__my_cfg_maps[item]
        return getattr(self.__parent(), item)

    def __getattr__(self, item):
        if item in self.__my_cfg_maps:
            return self.__my_cfg_maps[item]
        return getattr(self.__parent(), item)


def build_cfg_monkey_patch(cfg_maps: dict) -> NoReturn:
    """
    This monkey patch overrides current_build_config() so no one can disobey you on build_config key-value pairs
    :params cfg_maps: Dict
    :return: Nothing
    """
    replaced_modules = []
    original_function = None
    copied_modules = sys.modules.copy()
    for module in copied_modules:
        if module == "tbe.tvm.build_module":
            original_function = getattr(copied_modules[module], "current_build_config")
            break
    for module in copied_modules:
        if hasattr(copied_modules[module], "current_build_config"):
            replaced_modules.append(module)

            def __stub():
                return StubBuildConfig(original_function, cfg_maps)

            setattr(copied_modules[module], "current_build_config", __stub)
    yield
    for module in replaced_modules:
        setattr(copied_modules[module], "current_build_config", original_function)
    yield


def dynamic_build_monkey_patch(dsl: ModuleType, schedule_count: list) -> NoReturn:
    """
    This monkey patch overrides te.lang.dynamic.build() so you can retrieve dynamic shape schedule count
    :param dsl: module of tbe.dsl
    :param schedule_count: list
    :return: Nothing
    """
    original_function = dsl.build

    def _stub(sch, *args, **kwargs):
        def __recursive_get_schedule_count(container):
            size = 0
            for unit in container:
                if isinstance(unit, Sequence):
                    size += __recursive_get_schedule_count(unit)
                else:
                    size += 1
            return size

        schedule_count[0] = __recursive_get_schedule_count(sch) if isinstance(sch, Sequence) \
            else 1 if sch is not None \
            else 0
        return original_function(sch, *args, **kwargs)

    dsl.build = _stub
    yield
    dsl.build = original_function
    yield


# noinspection PyUnresolvedReferences
def rl_bank_monkey_patch(rl_bank: ModuleType, rl_return: list) -> NoReturn:
    """
    This monkey patch overrides te.domain.rl_bank.rl_bank.query_rl_bank so you can retrieve static rl_bank query result
    :param rl_bank: module of te.domain.rl_bank
    :param rl_return: list
    :return: Nothing
    """
    # noinspection PyUnresolvedReferences
    original_query_rl_bank = rl_bank.rl_bank.query_rl_bank

    def _stub(*args, **kwargs):
        result, sch = original_query_rl_bank(*args, **kwargs)
        rl_return[0] = str(result)
        return result, sch

    rl_bank.rl_bank.query_rl_bank = _stub
    yield
    rl_bank.rl_bank.query_rl_bank = original_query_rl_bank
    yield


def op_pattern_monkey_patch(cce_schedule: ModuleType, pattern: list) -> NoReturn:
    """
    This monkey patch overrides tbe.dsl.static_schedule.cce_schedule.global_core_schedule so you can retrieve schedule
    :param cce_schedule: module of te.dsl.static_schedule.cce_schedule
    :param pattern: list
    :return:
    """
    # noinspection PyUnresolvedReferences
    original_global_core_schedule = cce_schedule.global_core_schedule

    def _stub(*args, **kwargs):
        op_info = kwargs["op_info"]
        if "pattern" in op_info:
            pattern[0] = op_info["pattern"].name
        if "sub_pattern" in op_info:
            pattern[0] += "+"
            pattern[0] += op_info["sub_pattern"].name
        if "type" in op_info:
            pattern[0] += "+"
            pattern[0] += op_info["type"].name
        return original_global_core_schedule(*args, **kwargs)

    cce_schedule.global_core_schedule = _stub
    yield
    cce_schedule.global_core_schedule = original_global_core_schedule
    yield


def ir_pass_monkey_patch(ir_pass: ModuleType) -> NoReturn:
    """
    This monkey patch overrides te.tvm.ir_pass.VerifyArgs PASS so that god-damn parallel_fatbin won't be effective
    """

    def _stub(*_, **__):
        return False

    original_func = ir_pass.VerifyArgs
    ir_pass.VerifyArgs = _stub
    yield
    ir_pass.VerifyArgs = original_func
    yield


def soc_spec_monkey_patch(field_dict) -> NoReturn:
    """
    This monkey patch overrides value returned by get_soc_spec
    """
    original_function = None
    imported_modules = sys.modules.copy()
    for name, module in imported_modules.items():
        if name == "tbe.common.platform":
            original_function = module.get_soc_spec
            break

    def _stub_func(key):
        if key in field_dict:
            return field_dict[key]
        return original_function(key)

    for name, module in imported_modules.items():
        if hasattr(module, "get_soc_spec") and name != "tbe.common.platform":
            module.get_soc_spec = _stub_func
    yield
    for name, module in imported_modules.items():
        if hasattr(module, "get_soc_spec") and name != "tbe.common.platform":
            module.get_soc_spec = original_function
    yield
