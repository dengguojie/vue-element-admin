#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
"""
Ascend910 Series PEM Model Parser
"""
# Standard Packages
import re
from enum import Enum

# Third-party Packages
from .classes import INSTR_COLOR_MAP
from .classes import DavinciV100PipelineType
from .Ascend910_CA import Ascend910_CA_ModelParser


class Ascend910_PEM_ModelParser(Ascend910_CA_ModelParser):
    # ICACHE
    pattern_read = re.compile(r"^\[(\D+)] \[(\d+)]: icache read address is (0x\w+), size is (0x\w+), status is (\w+)$")
    pattern_request = re.compile(r"^\[(\D+)] \[(\d+)]: icache refill (\w+), id is (0x\w+), address is (0x\w+)$")
    pattern_special = re.compile(r"^\[(\D+)] \+{22}(.+)\+{22}\.$")
    pattern_end = re.compile(r"^core: (\d+), tick: (\d+), kernal(\d+) blkID (\d+) of blkdim (\d+) done$")
    # INSTR
    pattern_instr_normal = re.compile(r"^\[(\D*)] \[(\d*)] \(PC: (\w*)\) (\w+)\s+: \(Binary: (\w*)\) (\w+)\s+(.+)$")

    # bar
    pattern_flag = re.compile(r"^PIPE:(\w+), TRIGGER PIPE:(\w+), FLAG ID:\d$")
    pattern_bar = re.compile(r"^PIPE:(\w+)$")

    pattern_none = re.compile(r"^$")
    special_rules = ((re.compile(r"ST_X|LD_X.*"),
                      pattern_none,
                      lambda search_result, e_search_result, pipeline, instr, extra_info:
                      DavinciV100PipelineType.PID_PIPELINE_SCALARLDST),
                     (re.compile(r"SET_FLAG"),
                      pattern_none,
                      lambda search_result, e_search_result, pipeline, instr, extra_info: None),
                     (re.compile(r"WAIT_FLAG"),
                      pattern_flag,
                      lambda search_result, e_search_result, pipeline, instr, extra_info:
                      DavinciV100PipelineType["PID_PIPELINE_" + e_search_result.group(2)] if e_search_result else
                      RuntimeError(f"Failed determine instruction pipeline {instr} with extra_info {extra_info}")),
                     (re.compile(r"BAR"),
                      pattern_bar,
                      lambda search_result, e_search_result, pipeline, instr, extra_info:
                      DavinciV100PipelineType.PID_PIPELINE_SCALAR if extra_info == "PIPE:ALL" else
                      DavinciV100PipelineType["PID_PIPELINE_" + e_search_result.group(1)] if e_search_result else
                      RuntimeError(f"Failed determine instruction pipeline {instr} with extra_info {extra_info}")))

    @staticmethod
    def get_color_by_instr_name(instr: str, pipeline_type: Enum):
        if "WAIT_FLAG" == instr or "BAR" == instr:
            return INSTR_COLOR_MAP[DavinciV100PipelineType.PID_PIPELINE_FLOWCTRL]
        return Ascend910_CA_ModelParser.get_color_by_instr_name(instr, pipeline_type)

    @staticmethod
    def get_instr_name_by_instr(instr: tuple):
        if "WAIT_FLAG" == instr[5]:
            return instr[5] + "_" + instr[6].split(",")[0].split(":")[-1].strip()
        return instr[5]

    def __init__(self, container):
        super().__init__(container)
        self.ignored_popped_instructions.add("BAR")
