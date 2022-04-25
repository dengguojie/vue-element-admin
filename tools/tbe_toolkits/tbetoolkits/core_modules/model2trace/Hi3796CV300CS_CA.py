#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
"""
Ascend910 Series PEM Model Parser
"""
# Standard Packages
import re
from pathlib import Path
from typing import List
from typing import Optional

# Third-party Packages
from . import common
from .classes import DavinciV200PipelineType


def special_pipe_transformation(_, e_search_result, __, instr, extra_info, group=2):
    if e_search_result is None:
        raise RuntimeError(f"Failed determine instruction pipeline {instr} with extra_info {extra_info}")
    waiting_pipe = e_search_result.group(group)
    if waiting_pipe == "L2":
        waiting_pipe = "MTE2"
    elif waiting_pipe == "L3":
        waiting_pipe = "MTE3"
    elif waiting_pipe == "L1":
        waiting_pipe = "MTE1"
    return DavinciV200PipelineType["PID_PIPELINE_" + waiting_pipe]


class Hi3796CV300CS_CA_ModelParser(common.BaseModelParser):
    # ICACHE
    pattern_read = re.compile(r"^\[(\D+)] \[(\d+)]: icache read address is (0x\w+), size is (0x\w+), status is (\w+)$")
    pattern_request = re.compile(r"^\[(\D+)] \[(\d+)]: icache refill (\w+), id is (0x\w+), address is (0x\w+)$")
    pattern_special = re.compile(r"^\[(\D+)] \+{22}(.+)\+{22}\.$")
    pattern_end = re.compile(r"^core: (\d+), tick: (\d+), kernal(\d+) blkID (\d+) of blkdim (\d+) done$")

    pattern_icache_useless = (re.compile(r"^\[info] \[\d+]: icache load data is 0x\w+$"),
                              re.compile(r"^\[info] \[\d+]: icache refill request skipped$"),
                              re.compile(
                                  r"^\[info] \[\d+]: ccu miss_predict flush icache cmbf, normal_reqs id is 0x\w+$"),
                              re.compile(
                                  r"^\[info] \[\d+]: ccu miss_predict flush icache pipeline, normal_reqs id is 0x\w+$"),
                              re.compile(r"^\[info] \[\d+]: icache refill request has same addr in CMBF, "
                                         r"req_type: PREFETCH, address is 0x\w+$"),
                              re.compile(r"^\[info] \[\d+]: ccu end_execute flush icache clean_req_lst, "
                                         r"normal_reqs id is 0x\w+$"),
                              re.compile(
                                  r"^\[info] hwts preload operation, tick: \d+, preload_num: 0x\w+, addr: 0x\w+$"))
    # INSTR
    pattern_instr_normal = re.compile(
        r"^\[(\w+)] \[(\d+)]\(PC: (0x\w+)\)\s?(\w*\s*\w*) : \(Binary: (0x\w+)\) (\w+)(.*)$")
    flag_pattern = re.compile(r"^\(pipe_type: (\w+), tigger_pipe: (\w+), event_id: \d\) $")
    bar_pattern = re.compile(r"^\(pipe_type: (\w+)\) $")
    pattern_none = re.compile(r"^$")
    special_rules = ((re.compile(r"scalar_ld|scalar_st.*"),
                      pattern_none,
                      lambda search_result, e_search_result, pipeline, instr, extra_info:
                      DavinciV200PipelineType.PID_PIPELINE_SCALARLDST),
                     (re.compile(r"set_flag"),
                      pattern_none,
                      lambda search_result, e_search_result, pipeline, instr, extra_info: None),
                     (re.compile(r"wait_flag"),
                      flag_pattern,
                      special_pipe_transformation),
                     (re.compile(r"barrier"),
                      bar_pattern,
                      lambda search_result, e_search_result, pipeline, instr, extra_info:
                      DavinciV200PipelineType.PID_PIPELINE_SCALAR if extra_info == "(pipe_type: ALL) " else
                      special_pipe_transformation(search_result, e_search_result, pipeline, instr, extra_info, 1)))

    def read_dumps(self, path: Path, dump_type: str = "icache", core_index: int = 0) -> List[str]:
        return self.v100_read_dumps(path, dump_type, core_index)

    def initialize_metadata(self):
        return self.handle_icache_metadata()

    def _try_read_icache_log(self, _log) -> bool:
        return self.handle_icache_read_log(self.pattern_read, self.pattern_icache_useless,
                                           _log)

    def _try_request_icache_log(self, _log, icache_pipelines, queued_logs):
        return self.handle_icache_request_log(self.pattern_request,
                                              _log, icache_pipelines, queued_logs)

    def _try_end_of_model_log(self, _log):
        return self.handle_end_of_model_log(self.pattern_special, self.pattern_end,
                                            _log)

    # Handle ICache log use default

    def _try_instr_log(self, _log: str) -> Optional[tuple]:
        if _log.endswith("poped from IQ ") and "lsu_mov_special_xn" not in _log:
            return ()
        return self.parse_normal_instr_log(self.pattern_instr_normal, _log, lambda p: p.split(" ")[0])

    def _determine_pipeline_type(self, pipeline, instr: str, extra_info):
        if pipeline == "L2":
            pipeline = "MTE2"
        elif pipeline == "L3":
            pipeline = "MTE3"
        elif pipeline == "L1":
            pipeline = "MTE1"
        return self.handle_pipeline_type(self.special_rules, pipeline, instr, extra_info)

    def handle_instr_log(self, raw_instr_logs, raw_instr_popped_logs):
        pipelines = (DavinciV200PipelineType.PID_PIPELINE_MTE3,
                     DavinciV200PipelineType.PID_PIPELINE_MTE2,
                     DavinciV200PipelineType.PID_PIPELINE_MTE1,
                     DavinciV200PipelineType.PID_PIPELINE_VEC0,
                     DavinciV200PipelineType.PID_PIPELINE_CUBE,
                     DavinciV200PipelineType.PID_PIPELINE_SCALAR,
                     DavinciV200PipelineType.PID_PIPELINE_SCALARLDST)
        self._handle_instr_log(raw_instr_logs, raw_instr_popped_logs, pipelines)

    @staticmethod
    def get_color_by_instr_name(instr: str, pipeline_type: common.Enum):
        if "wait_flag" == instr or "barrier" == instr:
            return common.INSTR_COLOR_MAP[DavinciV200PipelineType.PID_PIPELINE_FLOWCTRL]
        return common.BaseModelParser.get_color_by_instr_name(instr, pipeline_type)

    @staticmethod
    def get_instr_name_by_instr(instr: tuple):
        if "wait_flag" == instr[5]:
            return instr[5] + "_" + instr[6].split(",")[0].split(":")[-1].strip()
        return instr[5]

    def __init__(self, container):
        super().__init__(container, DavinciV200PipelineType)
