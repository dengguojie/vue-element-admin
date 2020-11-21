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
Basic structure of schedule factory for all pure vector operator
Inherit VectorSchedule() and implement all abstract method to build your own schedule
"""

# Standard Packages
from abc import ABC
from enum import auto
from enum import Enum
from typing import Any
from typing import Set
from typing import List
from typing import Dict
from typing import Tuple
from typing import Union
from typing import Callable
from typing import Hashable
from typing import Iterable
from typing import NoReturn
from typing import Optional

# Third_party Packages
from te import tvm
from te.tvm.expr import Reduce
from te.tvm.tensor import Tensor
from te.tvm.schedule import Fuse
from te.tvm.schedule import Split
from te.tvm.schedule import Stage
from te.tvm.schedule import IterVar
from te.tvm.schedule import Schedule
from .util import is_keepdims
from .util import is_reduce_tensor
from .util import get_reduce_all_axes
from .util import get_reduce_axis_indices
from .vector_info import ComputeGraphInfo
from .vector_schedule_base import VectorScheduleBase


class VectorSchedule(VectorScheduleBase, ABC):
    class TilingInfo:
        class TilingMode(Enum):
            DEFAULT = "Default"
            FACTOR = "Factor"
            NPARTS = "Nparts"
            FUSE = "Fuse"
            RFACTOR = "RFactor"

        def __init__(self,
                     tensor: Union[Tensor, "VectorSchedule.Placeholder"] = None,
                     mode: TilingMode = TilingMode.DEFAULT,
                     axis_index: Union[int, "VectorSchedule.Placeholder"] = None,
                     factor: int = None,
                     use_inner_if_split: bool = True,
                     fuse_list: List[Union[int, "VectorSchedule.Placeholder"]] = None,
                     rfactor_scope: str = None):
            self.tiling_tensor: Union[Tensor, "VectorSchedule.Placeholder"] = tensor
            self.tiling_mode: VectorSchedule.TilingInfo.TilingMode = mode
            self.tiling_axis_index: Union[int, "VectorSchedule.Placeholder"] = axis_index
            self.factor: int = factor
            self.use_inner_if_split: bool = use_inner_if_split
            self.fuse_list: List[Union[int, "VectorSchedule.Placeholder"]] = fuse_list
            self.rfactor_scope: str = rfactor_scope

    class TilingResult:
        def __init__(self,
                     tiling_info: "VectorSchedule.TilingInfo",
                     result: Union[IterVar, Iterable[IterVar], Tensor]):
            self.tiling_info: "VectorSchedule.TilingInfo" = tiling_info
            if isinstance(result, Iterable):
                result = tuple(result)
            self.result: Union[Tuple[IterVar], IterVar, Tensor] = result

    class Placeholder(Hashable):
        class PlaceholderType(Enum):
            CACHE_READ_TENSOR = auto()
            CACHE_WRITE_TENSOR = auto()
            RFACTOR_TENSOR = auto()
            TILING_OUTER = auto()
            TILING_INNER = auto()
            FUSE_RESULT = auto()
            REDUCE_AXIS = auto()

        def __init__(self, _type: PlaceholderType, _key: Any):
            self.type = _type
            self.key = _key

        def __getattr__(self, item: str):
            if self.type == self.PlaceholderType.CACHE_READ_TENSOR or \
                    self.type == self.PlaceholderType.CACHE_WRITE_TENSOR:
                return getattr(self.key[0], item)
            elif self.type == self.PlaceholderType.RFACTOR_TENSOR:
                return getattr(self.key[1], item)
            else:
                raise KeyError("%s does not exist as a valid attribute for %s Placeholder" % (item, self.type))

        def __hash__(self):
            return hash((self.type, self.key))

        def __eq__(self, other):
            return type(other) == type(self) and other.type == self.type and other.key == self.key

        def __ne__(self, other):
            return type(other) != type(self) or other.type != self.type or other.key != self.key

        def __repr__(self):
            return "Placeholder|||%s of %s|||" % (str(self.type.name), str(self.key))

    class StorageAlignInfo:
        def __init__(self,
                     tensor: Union[Tensor, "VectorSchedule.Placeholder"] = None,
                     axis_idx: Union[int, "VectorSchedule.Placeholder"] = None,
                     factor: int = None,
                     offset: int = 0):
            self.tensor = tensor
            self.axis_index = axis_idx
            self.factor = factor
            self.offset = offset

    class EmitInsnInfo:
        def __init__(self,
                     tensor: Union[Tensor, "VectorSchedule.Placeholder"] = None,
                     axis_idx: Union[int, "VectorSchedule.Placeholder"] = None,
                     insn: str = None):
            self.tensor = tensor
            self.axis_index = axis_idx
            self.insn = insn

    def __init__(self,
                 graph_info: ComputeGraphInfo,
                 scope: str = "local.UB"):
        super().__init__()
        self.compute_scope: str = scope
        self.graph_info: ComputeGraphInfo = graph_info
        self.forward_compute_graph_map: Dict[Tensor, Set[Tensor]] = self.graph_info.tensor_consumers_map
        self.backward_compute_graph_map: Dict[Tensor, Set[Tensor]] = self.graph_info.tensor_producers_map
        self.forward_stage_graph_map: Dict[Tensor, Set[Tensor]] = \
            ComputeGraphInfo.set_map_deepcopy(self.forward_compute_graph_map)
        self.backward_stage_graph_map: Dict[Tensor, Set[Tensor]] = \
            ComputeGraphInfo.set_map_deepcopy(self.backward_compute_graph_map)
        self.tensor_reduced_axis_indices: Dict[Union[Tensor, VectorSchedule.Placeholder], List[int]] = {}
        # Data Flow Control - Calculate and Do
        self._output_scope: str = ""
        self._tensor_to_scope_map: Dict[Tensor, str] = {}
        self._data_flow_control: Dict[Union[VectorSchedule.Placeholder, Tensor],
                                      Dict[str,
                                           Set[Optional[Tuple[Union[Tensor,
                                                                    VectorSchedule.Placeholder], ...], ...]]]] = {}
        self._data_flow_control_placeholder_map: Dict[Tuple[Union[VectorSchedule.Placeholder, Tensor],
                                                            str,
                                                            Optional[Tuple[Union[Tensor,
                                                                                 VectorSchedule.Placeholder]], ...]],
                                                      Union[VectorSchedule.Placeholder, Tensor]] = {}
        # Compute inline - Calculate
        self.compute_inline_tensor_set: Set[Union[Tensor, VectorSchedule.Placeholder]] = set()
        # Storage Bound - Calculate
        self.storage_bound_map: Dict[Union[Tensor, VectorSchedule.Placeholder], int] = {}
        # Tiling - Calculate
        self._default_tiling_mode = VectorSchedule.TilingInfo.TilingMode.FACTOR
        self._tiling: List[Union[VectorSchedule.TilingInfo], ...] = []
        self._tensor_to_tiling_map: Dict[Tensor, Dict[int, List[VectorSchedule.TilingInfo, ...]]] = {}
        self._rfactor_map: Dict[Union[Tensor, VectorSchedule.Placeholder],
                                Dict[Union[IterVar, VectorSchedule.Placeholder],
                                     VectorSchedule.Placeholder]] = {}
        # Tiling - Do
        self._tiling_result_list: List[Union[VectorSchedule.TilingResult], ...] = []
        self._axis_to_tiling_result_map: Dict[IterVar, VectorSchedule.TilingResult] = {}
        # Reorder - Calculate
        self._tensor_to_reorder_map = {}
        # Constraint - Calculate
        self.constraint_func_pair_list = []
        # Storage Align - Calculate
        self.storage_align_list = []
        # Multi Core - Calculate
        self.multi_core_bind_tensor = None
        self.multi_core_bind_axis = None
        # ComputeAt - Calculate
        self.anchor_point_list = []
        self.anchor_point_axis_index_list = []
        # ComputeAt - Do
        self.compute_at_map: Dict[Tensor, Tensor] = {}
        # EmitInsn - Calculate
        self.emit_insn_list: List[VectorSchedule.EmitInsnInfo] = []

    def _do_create_schedule(self) -> NoReturn:
        self.schedule: Schedule = tvm.create_schedule([tensor.op for tensor in self.graph_info.output_tensor_set])

    def _do_data_flow_control(self) -> NoReturn:
        self._do_set_scope()
        self._do_cache_read_and_cache_write()

    def _do_set_scope(self) -> NoReturn:
        for tensor in self._tensor_to_scope_map:
            stage: Stage = self.schedule[tensor]
            scope: str = self._tensor_to_scope_map[tensor]
            stage.set_scope(scope)

    def _do_cache_read_and_cache_write(self) -> NoReturn:
        for tensor, tasks in self._data_flow_control.items():
            for source_buffer_scope, consumers_set in tasks.items():
                for consumers in consumers_set:
                    if consumers is None:
                        result: Tensor = self.schedule.cache_write(tensor, source_buffer_scope)
                        self._data_flow_control_placeholder_map[(tensor, source_buffer_scope, None)] = result
                    else:
                        list_consumers = list(consumers)
                        for idx, consumer in enumerate(list_consumers):
                            if isinstance(consumer, VectorSchedule.Placeholder):
                                list_consumers[idx] = self.solve_placeholder(consumer)
                        result: Tensor = self.schedule.cache_read(tensor, source_buffer_scope, list_consumers)
                        self._data_flow_control_placeholder_map[(tensor, source_buffer_scope, consumers)] = result

    def _do_compute_inline(self):
        for tensor in self.compute_inline_tensor_set:
            stage: Stage = self.schedule[self.solve_placeholder(tensor)]
            stage.compute_inline()

    def _do_storage_bound(self):
        for tensor, storage_bound in self.storage_bound_map.items():
            stage: Stage = self.schedule[self.solve_placeholder(tensor)]
            stage.set_storage_bound(storage_bound)

    def _do_tiling(self):
        def get_normal_tiling_parameters(_tiling: VectorSchedule.TilingInfo):
            tiling_tensor = self.solve_placeholder(_tiling.tiling_tensor)
            tiling_factor = _tiling.factor
            tiling_axis_index = _tiling.tiling_axis_index
            tiling_axis = self.get_itervar_by_original_index(_tiling.tiling_tensor, tiling_axis_index)
            use_inner = _tiling.use_inner_if_split
            while tiling_axis in self._axis_to_tiling_result_map:
                pre_tiling_result: VectorSchedule.TilingResult = self._axis_to_tiling_result_map[tiling_axis]
                if isinstance(pre_tiling_result.result, tuple):
                    if use_inner:
                        tiling_axis = pre_tiling_result.result[2]
                    else:
                        tiling_axis = pre_tiling_result.result[1]
                elif isinstance(pre_tiling_result.result, IterVar):
                    tiling_axis = pre_tiling_result.result
                else:
                    raise RuntimeError("Unsupported tiling for %s of type %s" % (str(pre_tiling_result.result),
                                                                                 str(type(pre_tiling_result.result))))
            return tiling_axis, tiling_factor, tiling_tensor

        def __handle_normal_tiling(_tiling: VectorSchedule.TilingInfo, use_factor: bool):
            tiling_axis, tiling_factor, tiling_tensor = get_normal_tiling_parameters(_tiling)
            if use_factor:
                outer, inner = self.schedule[tiling_tensor].split(tiling_axis,
                                                                  factor=tiling_factor)
            else:
                outer, inner = self.schedule[tiling_tensor].split(tiling_axis,
                                                                  nparts=tiling_factor)
            tiling_result = VectorSchedule.TilingResult(_tiling, (tiling_axis, outer, inner))
            self._tiling_result_list.append(tiling_result)
            self._axis_to_tiling_result_map[tiling_axis] = tiling_result

        def __handle_factor_tiling(_tiling: VectorSchedule.TilingInfo):
            __handle_normal_tiling(_tiling, True)

        def __handle_nparts_tiling(_tiling: VectorSchedule.TilingInfo):
            __handle_normal_tiling(_tiling, False)

        def __handle_fuse_tiling(_tiling: VectorSchedule.TilingInfo):
            fuse_tensor: Tensor = self.solve_placeholder(_tiling.tiling_tensor)
            fuse_stage: Stage = self.schedule[fuse_tensor]
            stage_used_axis_to_relation_map = self._get_stage_axis_to_relation_map(fuse_tensor)
            real_fuse_targets = [
                self.solve_placeholder(target) if isinstance(target, VectorSchedule.Placeholder) else
                self.get_itervar_by_original_index(_tiling.tiling_tensor, target)
                for target in _tiling.fuse_list]
            real_fuse_target = real_fuse_targets[0]
            if real_fuse_target in stage_used_axis_to_relation_map:
                relation = stage_used_axis_to_relation_map[real_fuse_target]
                if isinstance(relation, Split):
                    real_fuse_target = relation.outer
                elif isinstance(relation, Fuse):
                    real_fuse_target = relation.fused
                else:
                    raise RuntimeError("Unknown relation %s" % relation)
            for next_fuse_target in reversed(real_fuse_targets[1:]):
                real_fuse_target = fuse_stage.fuse(real_fuse_target, next_fuse_target)
            tiling_result = VectorSchedule.TilingResult(_tiling, real_fuse_target)
            self._tiling_result_list.append(tiling_result)
            for original_axis in real_fuse_targets:
                self._axis_to_tiling_result_map[original_axis] = tiling_result

        def __handle_rfactor_tiling(_tiling: VectorSchedule.TilingInfo):
            rfactor_tensor: Tensor = self.solve_placeholder(_tiling.tiling_tensor)
            rfactor_axis = self.solve_placeholder(_tiling.tiling_axis_index)
            rfactor_scope = _tiling.rfactor_scope
            if isinstance(rfactor_axis, int):
                rfactor_axis: IterVar = self.get_itervar_by_original_index(_tiling.tiling_tensor, rfactor_axis)
            result = self.schedule.rfactor(rfactor_tensor, rfactor_axis, _tiling.factor)
            self.schedule[result].set_scope(rfactor_scope)
            tiling_result = VectorSchedule.TilingResult(_tiling, result)
            self._tiling_result_list.append(tiling_result)

        for tiling in self._tiling:
            tiling_mode = tiling.tiling_mode
            if tiling_mode == VectorSchedule.TilingInfo.TilingMode.DEFAULT:
                tiling_mode = self._default_tiling_mode
            tiling_method_registry: Dict[VectorSchedule.TilingInfo.TilingMode, Callable] = \
                {
                    VectorSchedule.TilingInfo.TilingMode.FACTOR: __handle_factor_tiling,
                    VectorSchedule.TilingInfo.TilingMode.NPARTS: __handle_nparts_tiling,
                    VectorSchedule.TilingInfo.TilingMode.FUSE: __handle_fuse_tiling,
                    VectorSchedule.TilingInfo.TilingMode.RFACTOR: __handle_rfactor_tiling,
                }
            if tiling_mode in tiling_method_registry:
                tiling_method_registry[tiling_mode](tiling)
            else:
                raise RuntimeError("Unsupported split mode: %s" % str(tiling_mode))
        if self.multi_core_bind_axis is not None and self.multi_core_bind_tensor is not None:
            block = tvm.thread_axis("blockIdx.x")
            bind_tensor = self.multi_core_bind_tensor
            bind_axis = self.solve_placeholder(self.multi_core_bind_axis)
            if isinstance(bind_axis, int):
                bind_axis = self.get_itervar_by_original_index(bind_tensor, bind_axis)
            self.schedule[bind_tensor].bind(bind_axis, block)

    def _do_reorder(self):
        for tensor, reorder_target in self._tensor_to_reorder_map.items():
            real_tensor: Tensor = self.solve_placeholder(tensor)
            real_stage: Stage = self.schedule[real_tensor]
            real_reorder_target = [self.get_itervar_by_original_index(tensor, axis_idx)
                                   if isinstance(axis_idx, int) else
                                   self.solve_placeholder(axis_idx) for axis_idx in reorder_target]
            real_stage.reorder(*real_reorder_target)

    def _do_constraint(self):
        for constraint_func_pair in self.constraint_func_pair_list:
            params = [self.solve_placeholder(param) for param in constraint_func_pair[0]]
            func = constraint_func_pair[1]
            constraint_func_result = func(*params)
            if isinstance(constraint_func_result, (tuple, list)):
                for constraint in constraint_func_result:
                    self.schedule.set_constraint(constraint)
            else:
                self.schedule.set_constraint(constraint_func_result)

    def _do_storage_align(self):
        for storage_align in self.storage_align_list:
            tensor = self.solve_placeholder(storage_align.tensor)
            axis = self.get_itervar_by_original_index(storage_align.tensor, storage_align.axis_index)
            factor = storage_align.factor
            offset = storage_align.offset
            stage: Stage = self.schedule[tensor]
            stage.storage_align(axis, factor, offset)

    def _do_compute_at(self):
        for idx, anchor_point in enumerate(self.anchor_point_list):
            all_producers = self.get_all_producer_stages(anchor_point)
            anchor_point = self.solve_placeholder(anchor_point)
            anchor_stage = self.schedule[anchor_point]
            stage_axis_to_relations_map = self._get_stage_axis_to_relation_map(anchor_point)
            anchor_axis_index = self.anchor_point_axis_index_list[idx]
            if isinstance(anchor_axis_index, VectorSchedule.Placeholder):
                anchor_axis = self.solve_placeholder(anchor_axis_index)
            elif isinstance(anchor_axis_index, int):
                anchor_axis = self.get_itervar_by_original_index(self.anchor_point_list[idx], anchor_axis_index)
            else:
                raise RuntimeError("Invalid Anchor point axis index %s for anchor point %d %s" % (anchor_axis_index,
                                                                                                  idx,
                                                                                                  anchor_point))
            while anchor_axis in stage_axis_to_relations_map:
                anchor_axis = stage_axis_to_relations_map[anchor_axis].fused
            for producer in all_producers:
                if producer not in self.compute_at_map:
                    producer_stage: Stage = self.schedule[self.solve_placeholder(producer)]
                    producer_stage.compute_at(anchor_stage, anchor_axis)
                    self.compute_at_map[producer] = anchor_point

    def _do_emit_insn(self):
        for emitinsninfo in self.emit_insn_list:
            tensor: Tensor = self.solve_placeholder(emitinsninfo.tensor)
            stage: Stage = self.schedule[tensor]
            emitinsn_itervar: IterVar = self.solve_placeholder(emitinsninfo.axis_index)
            if isinstance(emitinsn_itervar, int):
                emitinsn_itervar = self.get_itervar_by_original_index(emitinsninfo.tensor, emitinsn_itervar)
            stage.emit_insn(emitinsn_itervar, emitinsninfo.insn)

    def _do_double_buffer(self):
        pass

    def solve_placeholder(self, placeholder: Any) -> Any:
        if not isinstance(placeholder, VectorSchedule.Placeholder):
            return placeholder

        def get_cache_read_or_write_buffer(key) -> Tensor:
            return self._data_flow_control_placeholder_map[key]

        def get_tiling_outer(key) -> IterVar:
            return self._tiling_result_list[key].result[1]

        def get_tiling_inner(key) -> IterVar:
            return self._tiling_result_list[key].result[2]

        def get_fuse_result(key) -> IterVar:
            return self._tiling_result_list[key].result

        def get_rfactor_result(key) -> Tensor:
            return self._tiling_result_list[key[0]].result

        def get_reduce_axis(key) -> IterVar:
            return self.schedule[self.solve_placeholder(key[0])].op.reduce_axis[key[1]]

        solution_registry: Dict[VectorSchedule.Placeholder.PlaceholderType, Callable] = {
            VectorSchedule.Placeholder.PlaceholderType.CACHE_READ_TENSOR: get_cache_read_or_write_buffer,
            VectorSchedule.Placeholder.PlaceholderType.CACHE_WRITE_TENSOR: get_cache_read_or_write_buffer,
            VectorSchedule.Placeholder.PlaceholderType.TILING_OUTER: get_tiling_outer,
            VectorSchedule.Placeholder.PlaceholderType.TILING_INNER: get_tiling_inner,
            VectorSchedule.Placeholder.PlaceholderType.FUSE_RESULT: get_fuse_result,
            VectorSchedule.Placeholder.PlaceholderType.RFACTOR_TENSOR: get_rfactor_result,
            VectorSchedule.Placeholder.PlaceholderType.REDUCE_AXIS: get_reduce_axis,
        }
        return solution_registry[placeholder.type](placeholder.key)

    def do_auto_data_flow_control(self, ignore_tensors: Tuple[Tensor] = ()) -> NoReturn:
        self._data_flow_control.clear()
        for tensor in self.forward_compute_graph_map:
            if tensor in ignore_tensors:
                continue
            my_scope: str = self._tensor_to_scope_map.setdefault(tensor, "")
            if tensor in self.graph_info.output_tensor_set:
                source_scope = None
                for producer in self.backward_compute_graph_map[tensor]:
                    producer_scope: str = self._tensor_to_scope_map.setdefault(producer, "")
                    if producer in self.graph_info.input_tensor_set:
                        producer_scope = self.compute_scope
                    if source_scope is None:
                        source_scope = producer_scope
                    if source_scope != producer_scope:
                        raise RuntimeError("Auto Data Flow Control failed: "
                                           "output tensor %s has multiple input scope %s vs %s." %
                                           (str(tensor), producer_scope, source_scope))
                if source_scope != my_scope:
                    self.cache_write(tensor, source_scope)
            scope_to_consumers_map: Dict[str, Set[Tensor, ...]] = {}
            for consumer in self.forward_compute_graph_map[tensor]:
                consumer_scope: str = self._tensor_to_scope_map.setdefault(consumer, "")
                if tensor in self.graph_info.input_tensor_set and consumer in self.graph_info.output_tensor_set:
                    consumer_scope = self.compute_scope
                    consumer = self.get_buffers_of(consumer)[0]
                different_scope = my_scope != consumer_scope
                not_mid_tensor = tensor not in self.graph_info.mid_tensor_set
                not_to_output_tensor = consumer not in self.graph_info.output_tensor_set
                not_mid_to_output_tensor = not_mid_tensor and not_to_output_tensor
                can_cache_read = different_scope and not_mid_to_output_tensor
                if can_cache_read:
                    scope_to_consumers_map.setdefault(consumer_scope, set()).add(consumer)
            for target_scope in scope_to_consumers_map:
                self.cache_read(tensor, target_scope, scope_to_consumers_map[target_scope])

    def cache_read(self,
                   source_tensor: Tensor,
                   target_scope: str,
                   consumers: Iterable[Tensor]) -> Placeholder:
        consumers = tuple(consumers)
        my_data_flow_control = self._data_flow_control.setdefault(source_tensor, {target_scope: {consumers, }})
        my_data_flow_control[target_scope].add(tuple(consumers))
        result_placeholder = VectorSchedule.Placeholder(VectorSchedule.Placeholder.PlaceholderType.CACHE_READ_TENSOR,
                                                        (source_tensor, target_scope, consumers))
        self._data_flow_control_placeholder_map[(source_tensor, target_scope, consumers)] = result_placeholder
        self.__stage_graph_manipulation(result_placeholder, source_tensor, False)
        if source_tensor in self.tensor_reduced_axis_indices:
            self.tensor_reduced_axis_indices[result_placeholder] = self.tensor_reduced_axis_indices[source_tensor]
        return result_placeholder

    def cache_write(self,
                    source_tensor: Tensor,
                    source_scope: str) -> Placeholder:
        my_data_flow_control = self._data_flow_control.setdefault(source_tensor, {source_scope: {None, }})
        my_data_flow_control[source_scope].add(None)
        result_placeholder = VectorSchedule.Placeholder(VectorSchedule.Placeholder.PlaceholderType.CACHE_WRITE_TENSOR,
                                                        (source_tensor, source_scope, None))
        self._data_flow_control_placeholder_map[(source_tensor, source_scope, None)] = result_placeholder
        self.__stage_graph_manipulation(result_placeholder, source_tensor, True)
        if source_tensor in self.tensor_reduced_axis_indices:
            if is_reduce_tensor(source_tensor) and not is_keepdims(source_tensor):
                for_original_tensor = self.tensor_reduced_axis_indices[source_tensor]
                self.tensor_reduced_axis_indices[source_tensor] = \
                    for_original_tensor + get_reduce_axis_indices(source_tensor)
                self.tensor_reduced_axis_indices[result_placeholder] = for_original_tensor
            else:
                self.tensor_reduced_axis_indices[result_placeholder] = self.tensor_reduced_axis_indices[source_tensor]
        return result_placeholder

    def get_buffers_of(self,
                       tensor: Tensor) -> List[Union[Placeholder, Tensor]]:
        placeholders = []
        for placeholder_idx, placeholder in self._data_flow_control_placeholder_map.items():
            if tensor == placeholder_idx[0]:
                placeholders.append(placeholder)
        if placeholders:
            return placeholders
        placeholders.append(tensor)
        return placeholders

    def get_all_producer_stages(self,
                                tensor: Union[Tensor, Placeholder]) -> Set[Union[Tensor, Placeholder]]:
        producers = set()
        for producer in self.backward_stage_graph_map[tensor]:
            producers.add(producer)
            producers.update(self.get_all_producer_stages(producer))
        return producers

    def get_all_consumer_stages(self,
                                tensor: Union[Tensor, Placeholder]) -> Set[Union[Tensor, Placeholder]]:
        consumers = set()
        for consumer in self.forward_stage_graph_map[tensor]:
            consumers.add(consumer)
            consumers.update(self.get_all_consumer_stages(consumer))
        return consumers

    def add_tiling(self,
                   tiling_tensor: Union[Placeholder, Tensor],
                   tiling_axis_index: Union[Placeholder, int] = None,
                   tiling_factor: Any = None,
                   tiling_mode: TilingInfo.TilingMode = TilingInfo.TilingMode.DEFAULT,
                   use_inner_if_split: bool = True,
                   fuse_axis_list: List[Union[int, TilingInfo]] = None,
                   rfactor_scope: str = None) -> int:
        tiling_info: VectorSchedule.TilingInfo = VectorSchedule.TilingInfo(tiling_tensor,
                                                                           tiling_mode,
                                                                           tiling_axis_index,
                                                                           tiling_factor,
                                                                           use_inner_if_split,
                                                                           fuse_axis_list,
                                                                           rfactor_scope)
        tiling_index = len(self._tiling)
        self._tiling.append(tiling_info)
        tiling_dict_of_tensor: Dict[int, List[VectorSchedule.TilingInfo, ...]] = \
            self._tensor_to_tiling_map.setdefault(tiling_tensor, {})
        tiling_list_of_axis: List[VectorSchedule.TilingInfo, ...] = \
            tiling_dict_of_tensor.setdefault(tiling_axis_index, [])
        tiling_list_of_axis.append(tiling_info)
        return tiling_index

    def rfactor(self,
                source_tensor: Tensor,
                factored_reduce_axis: Union[Placeholder, int],
                axis_factor: int = 0,
                scope: str = "special_rfactor") -> Placeholder:
        tiling_index = self.add_tiling(source_tensor,
                                       factored_reduce_axis,
                                       axis_factor,
                                       VectorSchedule.TilingInfo.TilingMode.RFACTOR,
                                       rfactor_scope=scope)
        placeholder = VectorSchedule.Placeholder(VectorSchedule.Placeholder.PlaceholderType.RFACTOR_TENSOR,
                                                 (tiling_index, source_tensor))
        self.__stage_graph_manipulation(placeholder, source_tensor, True)
        if source_tensor in self.tensor_reduced_axis_indices:
            self.tensor_reduced_axis_indices[placeholder] = self.tensor_reduced_axis_indices[source_tensor]
        return placeholder

    def split(self,
              source_tensor: Tensor,
              split_axis_index: Union[Placeholder, int],
              factor: int = None,
              nparts: int = None) -> Tuple[Placeholder, Placeholder]:
        if factor is not None and nparts is not None:
            raise RuntimeError("Cannot use factor mode and nparts mode at the same time!")
        if factor is None and nparts is None:
            raise RuntimeError("Must select one of factor mode or nparts mode!")
        if factor is not None:
            tiling_index = self.add_tiling(source_tensor, split_axis_index, factor,
                                           VectorSchedule.TilingInfo.TilingMode.FACTOR)
        else:
            tiling_index = self.add_tiling(source_tensor, split_axis_index, factor,
                                           VectorSchedule.TilingInfo.TilingMode.NPARTS)
        return (VectorSchedule.Placeholder(VectorSchedule.Placeholder.PlaceholderType.TILING_OUTER, tiling_index),
                VectorSchedule.Placeholder(VectorSchedule.Placeholder.PlaceholderType.TILING_INNER, tiling_index))

    def fuse(self,
             source_tensor: Tensor,
             fuse_axis_list: List[Union[int, Placeholder]]) -> Placeholder:
        tiling_index = self.add_tiling(source_tensor,
                                       tiling_mode=VectorSchedule.TilingInfo.TilingMode.FUSE,
                                       fuse_axis_list=fuse_axis_list)
        return VectorSchedule.Placeholder(VectorSchedule.Placeholder.PlaceholderType.FUSE_RESULT, tiling_index)

    def is_tiling_axis_index(self,
                             tensor: Union[Placeholder, Tensor],
                             axis_index: int) -> bool:
        if tensor in self._tensor_to_tiling_map and axis_index in self._tensor_to_tiling_map[tensor]:
            return True
        return False

    def get_itervar_by_original_index(self,
                                      tensor: Union[Placeholder, Tensor], axis_index: int) -> IterVar:
        real_tensor: Tensor = self.solve_placeholder(tensor)
        if tensor not in self.tensor_reduced_axis_indices:
            raise NotImplementedError("Please maintain reduce_axis_indices info for tensor %s" % (str(real_tensor)))
        reduced_indices = self.tensor_reduced_axis_indices[tensor]
        real_stage: Stage = self.schedule[real_tensor]
        body = real_tensor.op.body[0]
        calibrated_index = axis_index
        for reduce_index in reduced_indices:
            if reduce_index < axis_index:
                calibrated_index -= 1
        if isinstance(body, Reduce) and real_tensor not in self.graph_info.output_tensor_set:
            reduce_axis_indices = get_reduce_axis_indices(tensor)
            iter_var = None
            if calibrated_index in reduce_axis_indices:
                iter_var = real_stage.all_iter_vars[reduce_axis_indices.index(calibrated_index) + len(tensor.shape)]
            else:
                all_axes_var = get_reduce_all_axes(tensor)
                var = all_axes_var[calibrated_index]
                for _iter_var in real_stage.all_iter_vars:
                    if str(var) in str(_iter_var.var):
                        iter_var = _iter_var
                        break
            if iter_var is None:
                raise RuntimeError("Cannot find true axis %d for tensor %s" % (axis_index, str(real_tensor)))
            return iter_var
        else:
            try:
                return real_stage.all_iter_vars[calibrated_index]
            except IndexError:
                raise IndexError("Possible victim of false keepdims, use reduced_indices to fix this")

    def _get_stage_axis_to_relation_map(self,
                                        tensor: Union[Tensor, Placeholder]) -> Dict[IterVar, Union[Split, Fuse]]:
        stage_relations = list(self.schedule[self.solve_placeholder(tensor)].relations)
        stage_used_axis_map: Dict[IterVar, Union[Split, Fuse]] = {}
        for relation in stage_relations:
            if isinstance(relation, Split):
                stage_used_axis_map[relation.parent] = relation
            elif isinstance(relation, Fuse):
                stage_used_axis_map[relation.outer] = relation
                stage_used_axis_map[relation.inner] = relation
            else:
                raise RuntimeError("Unknown relation type %s" % relation)
        return stage_used_axis_map

    def __stage_graph_manipulation(self,
                                   new_tensor: Union[Placeholder, Tensor],
                                   related_tensor: Tensor,
                                   before: bool):
        if before:
            self.forward_stage_graph_map.setdefault(new_tensor, set())
            self.backward_stage_graph_map.setdefault(new_tensor, set())
            for producer in tuple(self.backward_stage_graph_map[related_tensor]):
                self.forward_stage_graph_map[producer].remove(related_tensor)
                self.forward_stage_graph_map[producer].add(new_tensor)
                self.backward_stage_graph_map[related_tensor].remove(producer)
                self.backward_stage_graph_map[new_tensor].add(producer)
            self.forward_stage_graph_map[new_tensor].add(related_tensor)
            self.backward_stage_graph_map[related_tensor].add(new_tensor)
        else:
            self.forward_stage_graph_map.setdefault(new_tensor, set())
            self.backward_stage_graph_map.setdefault(new_tensor, set())
            for consumer in tuple(self.forward_stage_graph_map[related_tensor]):
                self.forward_stage_graph_map[related_tensor].discard(consumer)
                self.backward_stage_graph_map[consumer].discard(related_tensor)
                self.backward_stage_graph_map[consumer].add(new_tensor)
                self.forward_stage_graph_map[new_tensor].add(consumer)
            self.forward_stage_graph_map[related_tensor].add(new_tensor)
            self.backward_stage_graph_map[new_tensor].add(related_tensor)
