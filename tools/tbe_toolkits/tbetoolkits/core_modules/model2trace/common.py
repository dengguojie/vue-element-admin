#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
"""
Common Functions
"""
# Standard Packages
import re
from abc import ABC
from abc import abstractmethod
from enum import Enum
from enum import EnumMeta
from pathlib import Path
from typing import Set
from typing import List
from typing import Dict
from typing import Tuple
from typing import NoReturn
from typing import Optional
from typing import Sequence
from typing import Callable

# Third-party Packages
from .classes import ChromeTraceJson
from .classes import ChromeInstantEventScope
from .classes import ChromeMetadataEvent
from .classes import ChromeInstantEvent
from .classes import ChromeCompleteEvent
from .classes import INSTR_COLOR_MAP
from .classes import ChromeColorName


class BaseModelParser(ABC):
    def __init__(self, container: ChromeTraceJson, pipeline_enum: EnumMeta):
        self.container: ChromeTraceJson = container
        self.ignored_popped_instructions: Set[str] = set()
        self.pipeline_enum = pipeline_enum
        self.icache_request_memory: Dict[str, set] = {}

    @staticmethod
    def get_pipeline_name(pipeline_type):
        return pipeline_type.name.split("_")[-1]

    @staticmethod
    def v100_read_dumps(path: Path, dump_type: str = "icache", core_index: int = 0) -> List[str]:
        rotating_indexes = []
        rotaing_file_name = f"core{core_index}_{dump_type}_log.*.dump"
        icache_dump_rotating_files = path.glob(rotaing_file_name)
        for f in icache_dump_rotating_files:
            f_groups = f.name.split('.')
            rotating_index = int(f_groups[-2])
            rotating_indexes.append(rotating_index)
        icache_dump_latest_file = path / f"core{core_index}_{dump_type}_log.dump"
        if icache_dump_latest_file.exists():
            rotating_indexes.append(0)
        raw_data = []
        for idx in rotating_indexes:
            print("Reading", path / f"core{core_index}_{dump_type}_log.{idx}.dump")
            with open((path / f"core{core_index}_{dump_type}_log.{idx}.dump") if idx > 0 else
                      (path / f"core{core_index}_{dump_type}_log.dump"),
                      encoding="UTF-8") as f:
                raw_data += f.read().splitlines()
        while raw_data and not raw_data[-1]:
            raw_data.pop()
        return raw_data

    def handle_pipeline_type(self, rules: Tuple[Tuple[re.Pattern, re.Pattern, Callable], ...],
                             pipeline: str, instr: str, extra_info: str):
        for pattern, extra_pattern, func in rules:
            match_pattern = pattern.search(instr)
            if match_pattern:
                res = func(match_pattern, extra_pattern.search(extra_info), pipeline, instr, extra_info)
                if isinstance(res, BaseException):
                    raise res
                return res
        return self.pipeline_enum["PID_PIPELINE_" + pipeline]

    def initialize_sub_pipelines(self, pipeline_enum_elements: Sequence[Enum]):
        subpipelines = {}
        for element in pipeline_enum_elements:
            subpipelines[element] = [False]
        for pipeline in pipeline_enum_elements:
            self.add_pipeline(pipeline.value, BaseModelParser.get_pipeline_name(pipeline))
        return subpipelines

    def add_pipeline(self, pipeline_id, pipeline_name):
        self.container.addEvent(ChromeMetadataEvent("process_name", pipeline_id, 0,
                                                    "name", str(pipeline_id).zfill(2) + "_" + pipeline_name))
        self.container.addEvent(ChromeMetadataEvent("thread_name", pipeline_id, 0,
                                                    "name",
                                                    str(pipeline_id).zfill(2) + "_" + pipeline_name + "_" + "00"))

    def add_sub_pipeline(self, pipeline_id, pipeline_name, sub_pipeline_id):
        self.container.addEvent(ChromeMetadataEvent("thread_name", pipeline_id, sub_pipeline_id,
                                                    "name", str(pipeline_id).zfill(2) + "_" + pipeline_name
                                                    + "_" + str(sub_pipeline_id).zfill(2)))

    def handle_icache_metadata(self):
        self.add_pipeline(self.pipeline_enum.PID_PIPELINE_ICACHE.value, "ICACHE")
        self.add_pipeline(self.pipeline_enum.PID_PIPELINE_ICACHEREAD.value, "ICACHELOAD")
        self.container.addEvent(ChromeCompleteEvent("ICache::Init", ("ICACHE",), 0,
                                                    self.pipeline_enum.PID_PIPELINE_ICACHEREAD.value, 0,
                                                    0))
        if not self.container.start_event:
            self.container.start_event = ChromeInstantEvent("Model::Start", ("Model",), 0,
                                                            self.pipeline_enum.PID_PIPELINE_ICACHE.value, 0,
                                                            ChromeInstantEventScope.GLOBAL,
                                                            cname=INSTR_COLOR_MAP[
                                                                self.pipeline_enum.PID_PIPELINE_ICACHE])

    def handle_icache_read_log(self, icache_read_pattern, useless_patterns, log):
        search_result = icache_read_pattern.search(log)
        if search_result:
            # ICache Read Operation
            # log_level = search_result.group(1)
            cycle = int(search_result.group(2))
            read_address = search_result.group(3)
            size = search_result.group(4)
            status = search_result.group(5)
            self.container.addEvent(ChromeInstantEvent("READ", ("ICACHEREAD",), cycle,
                                                       self.pipeline_enum.PID_PIPELINE_ICACHEREAD.value, 0,
                                                       ChromeInstantEventScope.PROCESS,
                                                       args={"address": read_address,
                                                             "size": size,
                                                             "status": status,
                                                             "cycle": cycle},
                                                       cname=ChromeColorName.GREY if status == "HIT"
                                                       else ChromeColorName.RED))
            return True
        if any(pattern.match(log) for pattern in useless_patterns):
            return True
        return False

    def handle_icache_request_log(self, pattern_request: re.Pattern, _log: str,
                                  icache_pipelines: list, queued_logs: dict,
                                  pattern_preload=None):
        search_result = pattern_request.search(_log)
        if not search_result and pattern_preload:
            search_result = pattern_preload.search(_log)
        if search_result:
            # ICache refill request Operation
            # log_level = search_result.group(1)
            cycle = int(search_result.group(2))
            request_stage = search_result.group(3)
            request_id = search_result.group(4)
            address = search_result.group(5)
            if request_stage == "request" or request_stage == "preload":
                # Push ICache
                if request_id in queued_logs:
                    # Another request
                    print(f"ICache request of id {request_id} duplicated! Use first met request instead.")
                else:
                    # First met
                    found_pipe = None
                    for pipe in icache_pipelines:
                        if not icache_pipelines[pipe]:
                            icache_pipelines[pipe] = True
                            found_pipe = pipe
                            break
                    if found_pipe is None:
                        self.add_sub_pipeline(self.pipeline_enum.PID_PIPELINE_ICACHE.value,
                                              "ICACHE", len(icache_pipelines))
                        found_pipe = len(icache_pipelines)
                        icache_pipelines[len(icache_pipelines)] = True
                    queued_logs[request_id] = [(cycle, address, found_pipe), set()]
                return True
            elif request_stage == "acknowledge":
                if request_id in queued_logs:
                    # Form full event
                    request_cycle = queued_logs[request_id][0][0]
                    request_address = queued_logs[request_id][0][1]
                    request_pipeline = queued_logs[request_id][0][2]
                    del queued_logs[request_id]
                    if request_address != address:
                        print(f"ICache refill request address not match for request id {request_id}: "
                              f"{request_address} vs {address}")
                    event = ChromeCompleteEvent("LOAD", ("ICACHE",), request_cycle,
                                                self.pipeline_enum.PID_PIPELINE_ICACHE.value, request_pipeline,
                                                cycle - request_cycle,
                                                args={
                                                    "address": address,
                                                    "request_id": request_id,
                                                    "cycle": request_cycle,
                                                    "duration": cycle - request_cycle,
                                                },
                                                cname=INSTR_COLOR_MAP[self.pipeline_enum.PID_PIPELINE_ICACHE])
                    if request_id not in self.icache_request_memory:
                        self.icache_request_memory[request_id] = set()
                    self.icache_request_memory[request_id].add(cycle)
                    self.container.addEvent(event)
                    icache_pipelines[request_pipeline] = False
                elif request_id in self.icache_request_memory and cycle in self.icache_request_memory[request_id]:
                    pass
                else:
                    # First met
                    print(f"ICache request anomaly encountered: "
                          f"acknowledge happened before request at cycle {cycle}, dropping.")
                return True
            else:
                raise RuntimeError(f"Unknown icache request at cycle {cycle}: {_log}")
        return False

    def handle_end_of_model_log(self, pattern_special: re.Pattern, pattern_end: re.Pattern,
                                _log: str) -> bool:
        search_result = pattern_special.search(_log)
        if search_result:
            _ = search_result.group(1)
            message = search_result.group(2)
            search_result = pattern_end.search(message)
            if search_result:
                # End of log
                core_num = search_result.group(1)
                total_tick = int(search_result.group(2))
                kernel_id = search_result.group(3)
                block_id = search_result.group(4)
                block_dim = search_result.group(5)
                if not self.container.end_event:
                    self.container.end_event = (ChromeInstantEvent("Model::End", ("Model",), total_tick,
                                                                   self.pipeline_enum.PID_PIPELINE_ICACHE.value, 0,
                                                                   ChromeInstantEventScope.GLOBAL,
                                                                   cname=ChromeColorName.BLACK,
                                                                   args={"core_num": core_num,
                                                                         "total_blocks": block_dim,
                                                                         "block_id": block_id,
                                                                         "kernel_id": kernel_id,
                                                                         "total_cycle": total_tick}))
                return True
        return False

    @staticmethod
    def parse_normal_instr_log(normal_pattern, log, pipeline_processing_func=None):
        search_result = normal_pattern.search(log)
        if search_result:
            # Normal ICache log
            _log_level = search_result.group(1)
            _cycle = int(search_result.group(2))
            _pc = search_result.group(3)
            if pipeline_processing_func:
                _pipeline = pipeline_processing_func(search_result.group(4))
            else:
                _pipeline = search_result.group(4)
            _binary = search_result.group(5)
            _instr = search_result.group(6)
            _extra_info = search_result.group(7)
            return _log_level, _cycle, _pc, _pipeline, _binary, _instr, _extra_info
        return None

    @staticmethod
    def instr_log_processing_unit_selection(instr_log_index, instr_logs,
                                            instr_popped_log_index, instr_popped_logs):
        is_start = False
        if instr_popped_log_index < len(instr_popped_logs):
            popped_cycle = instr_popped_logs[instr_popped_log_index][1]
            end_cycle = instr_logs[instr_log_index][1]
            if popped_cycle < end_cycle:
                is_start = True
                process_log = instr_popped_logs[instr_popped_log_index]
            elif popped_cycle == end_cycle:
                is_start = None
                process_log = ([], [])
                for i in range(instr_popped_log_index, len(instr_popped_logs)):
                    if instr_popped_logs[i][1] == popped_cycle:
                        process_log[0].append(instr_popped_logs[i])
                    else:
                        break
                for i in range(instr_log_index, len(instr_logs)):
                    if instr_logs[i][1] == end_cycle:
                        process_log[1].append(instr_logs[i])
                    else:
                        break
            else:
                process_log = instr_logs[instr_log_index]
        else:
            process_log = instr_logs[instr_log_index]
        return is_start, process_log

    def looking_for_sub_pipeline(self, my_sub_pipeline, pipeline_type):
        locked_sub_pipeline = None
        for idx, pipe in enumerate(my_sub_pipeline):
            if not pipe:
                my_sub_pipeline[idx] = True
                locked_sub_pipeline = idx
                break
        if locked_sub_pipeline is None:
            my_sub_pipeline.append(True)
            locked_sub_pipeline = len(my_sub_pipeline) - 1
            self.add_sub_pipeline(pipeline_type.value, pipeline_type.name.split('_')[-1],
                                  len(my_sub_pipeline) - 1)
        return locked_sub_pipeline

    @staticmethod
    def get_color_by_instr_name(instr: str, pipeline_type: Enum):
        return INSTR_COLOR_MAP[pipeline_type]

    @staticmethod
    def get_instr_name_by_instr(instr: tuple):
        return instr[5]

    @abstractmethod
    def _determine_pipeline_type(self, pipeline, instr: str, extra_info) -> Enum:
        """"""

    @abstractmethod
    def read_dumps(self, path: Path, dump_type: str = "icache", core_index: int = 0) -> List[str]:
        """"""

    @abstractmethod
    def initialize_metadata(self):
        """"""

    @abstractmethod
    def _try_read_icache_log(self, log: str):
        """"""

    @abstractmethod
    def _try_request_icache_log(self, log, icache_pipelines, queued_logs):
        """"""

    @abstractmethod
    def _try_end_of_model_log(self, log):
        """"""

    def handle_icache_log(self, icache_logs):
        queued_logs: Dict[str, List[tuple]] = {}
        icache_pipelines: Dict[int, bool] = {}
        for log in icache_logs:
            if not self._try_read_icache_log(log):
                if not self._try_request_icache_log(log, icache_pipelines, queued_logs):
                    if not self._try_end_of_model_log(log):
                        print(f"Could not identify ICACHE log: {log}")

    @abstractmethod
    def _try_instr_log(self, _log: str) -> Optional[tuple]:
        """Return None if not match, () if ignored"""

    def _instr_pre_parsing(self, raw_instr_logs):
        instr_logs = []
        for log in raw_instr_logs:
            instr_result = self._try_instr_log(log)
            if instr_result:
                instr_logs.append(instr_result)
                continue
            if instr_result == ():
                continue
            if self._try_end_of_model_log(log):
                continue
            print(f"Could not identify instr log: {log}")
        return instr_logs

    def _process_popped_instr(self, instr,
                              running_tasks,
                              sub_pipelines):
        log_level, cycle, pc, pipeline, binary, instr, extra_info = instr
        pipeline_type = self._determine_pipeline_type(pipeline, instr, extra_info)
        if pipeline_type is None:
            return
        if instr in self.ignored_popped_instructions:
            pass
        else:
            my_sub_pipeline = sub_pipelines[pipeline_type]
            locked_sub_pipeline = self.looking_for_sub_pipeline(my_sub_pipeline, pipeline_type)
            if pc not in running_tasks:
                running_tasks[pc] = [(cycle, pc, pipeline, binary, instr, extra_info, locked_sub_pipeline)]
            else:
                running_tasks[pc].append((cycle, pc, pipeline, binary, instr, extra_info, locked_sub_pipeline))

    def _process_instr(self, _instr, running_tasks, sub_pipelines) -> NoReturn:
        log_level, cycle, pc, pipeline, binary, instr, extra_info = _instr
        pipeline_type = self._determine_pipeline_type(pipeline, instr, extra_info)
        if pipeline_type is None:
            return
        if pc not in running_tasks:
            self.container.addEvent(ChromeInstantEvent(self.get_instr_name_by_instr(_instr),
                                                       (pipeline_type.name,),
                                                       cycle,
                                                       pipeline_type.value, 0,
                                                       ChromeInstantEventScope.PROCESS,
                                                       cname=self.get_color_by_instr_name(instr, pipeline_type),
                                                       args={"pc": pc,
                                                             "start": cycle,
                                                             "binary": binary,
                                                             "extra_info": extra_info}))
        else:
            st_cycle, st_pc, st_pipeline, st_binary, st_instr, st_extra_info, st_locked_pipe = \
                running_tasks[pc].pop(0)
            if not running_tasks[pc]:
                del running_tasks[pc]
            my_color = self.get_color_by_instr_name(instr, pipeline_type)
            self.container.addEvent(
                ChromeCompleteEvent(f"{self.get_instr_name_by_instr(_instr)}",
                                    (self.get_pipeline_name(pipeline_type),),
                                    st_cycle,
                                    pipeline_type.value, st_locked_pipe,
                                    cycle - st_cycle,
                                    args={"pc": st_pc, "start": st_cycle, "end": cycle, "duration": cycle - st_cycle,
                                          "binary": binary, "extra_info": extra_info},
                                    cname=my_color))
            sub_pipelines[pipeline_type][st_locked_pipe] = False

    def _process_multi_instr(self,
                             process_log,
                             running_tasks, sub_pipelines):
        for popped_idx, popped_instr in enumerate(process_log[0]):
            log_level, cycle, pc, pipeline, binary, instr, extra_info = popped_instr
            for idx, instr in enumerate(process_log[1]):
                if instr is not None:
                    _, _, end_pc, _, _, _, _ = instr
                    if pc == end_pc:
                        self._process_instr(instr,
                                            running_tasks, sub_pipelines)
                        process_log[0][popped_idx] = None
                        process_log[1][idx] = None
        for instr in process_log[1]:
            if instr is not None:
                self._process_instr(instr,
                                    running_tasks, sub_pipelines)
        for instr in process_log[0]:
            if instr is not None:
                self._process_popped_instr(instr,
                                           running_tasks, sub_pipelines)

    def _parse_instructions(self, instr_logs, instr_popped_logs, sub_pipelines):
        instr_log_index = 0
        instr_popped_log_index = 0
        running_tasks = {}
        while instr_log_index < len(instr_logs):
            # Select current processing log from start or end
            is_start, process_log = self.instr_log_processing_unit_selection(instr_log_index, instr_logs,
                                                                             instr_popped_log_index, instr_popped_logs)
            try:
                # Check for selection result
                if is_start is not None:
                    if is_start:
                        self._process_popped_instr(process_log,
                                                   running_tasks, sub_pipelines)
                        instr_popped_log_index += 1
                    else:
                        self._process_instr(process_log,
                                            running_tasks, sub_pipelines)
                        instr_log_index += 1
                else:
                    self._process_multi_instr(process_log,
                                              running_tasks, sub_pipelines)
                    instr_popped_log_index += len(process_log[0])
                    instr_log_index += len(process_log[1])
            except:
                print("Failed processing instruction", process_log)
                raise
        if instr_popped_log_index < len(instr_popped_logs):
            print(f"Popped log stopped responding at {instr_popped_log_index}, actual ending: {len(instr_popped_logs)}")
        if len(running_tasks):
            print(f"Found {len(running_tasks)} never-ending instructions")
            print(running_tasks)

    def _handle_instr_log(self, raw_instr_logs, raw_instr_popped_logs, pipelines):
        instr_logs = self._instr_pre_parsing(raw_instr_logs)
        instr_popped_logs = self._instr_pre_parsing(raw_instr_popped_logs)

        sub_pipelines = self.initialize_sub_pipelines(pipelines)
        if instr_logs and instr_popped_logs:
            self._parse_instructions(instr_logs,
                                     instr_popped_logs, sub_pipelines)
        else:
            print("Instruction log not found, skipping.")

    @abstractmethod
    def handle_instr_log(self, raw_instr_logs, raw_instr_popped_logs):
        """"""
