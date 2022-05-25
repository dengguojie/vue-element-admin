#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Copyright 2019-2021 Huawei Technologies Co., Ltd
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
Schedule primitive enhancement plug-in
"""
# Standard Packages
from typing import Set
from typing import Dict
from typing import List
from typing import Tuple
from typing import AnyStr
from typing import Mapping
from typing import Iterable
from typing import Optional
from enum import Enum
from enum import auto
# Ascend Packages
from tbe import tvm
from tbe.common.utils.errormgr import get_error_message


def _raise_error(message: AnyStr):
    """
    Raise error
    @param message:
    @return:
    """
    dict_args = {"errCode": "E90001", "detailed_cause": message}
    raise RuntimeError(dict_args, get_error_message(dict_args))


class TensorType(Enum):
    """
    Enum for Tensor Types
    """
    ANY = auto()
    ELEMENTWISE = auto()
    BROADCAST = auto()
    REDUCE = auto()


class IterType(Enum):
    """
    Iter Var Type
    """
    kDataPar = 0 # Data parallel iteration.
    kThreadIndex = 1 # The IterVar itself is a thread-index of a fixed thread launching group.
    kCommReduce = 2 # Communicative reduction.
    kOrdered = 3 # Serial loops with loop carry dependency, the iteration must execute in order. Cannot be re-ordered.
    kOpaque = 4 # IterVar is opaque


class Schedule:
    """
    Schedule Toolbox
    """

    def __init__(self, outs: List[tvm.tensor.Tensor] or Tuple[tvm.tensor.Tensor]):
        self.outs: List[tvm.tensor.Tensor] = outs
        self.sch: tvm.schedule.Schedule = self.create_schedule(outs)
        self.ori_tensors: List[tvm.tensor.Tensor] = self.tensors

        self.stages_not_on_ub: Set[tvm.schedule.Stage] = set()
        self.cache_read_stages: Dict[tvm.schedule.Stage, tvm.tensor.Tensor] = {}
        self.cache_write_stages: Dict[tvm.schedule.Stage, tvm.tensor.Tensor] = {}
    
    def __getitem__(self, stage: tvm.schedule.Stage):
        return self.sch[stage]
    
    @property
    def stages(self) -> List[tvm.schedule.Stage]:
        return self.sch.stages
    
    @property
    def stage_map(self) -> tvm.container.Map or Mapping[tvm.tensor.ComputeOp or tvm.tensor.PlaceholderOp,
                                                        tvm.schedule.Stage]:
        return self.sch.stage_map
    
    @property
    def outputs(self) -> List[tvm.tensor.ComputeOp]:
        return self.sch.outputs
    
    @property
    def real_outputs(self) -> List[tvm.tensor.ComputeOp]:
        return list(set(self.outputs))
    
    @property
    def tensors(self) -> List[tvm.tensor.Tensor]:
        tensors = set(self.outs)
        for stage in self.stages:
            for tensor in stage.op.input_tensors:
                tensors.add(tensor)
        return list(tensors)
    
    @property
    def placeholder(self) -> List[tvm.tensor.Tensor]:
        return [tensor for tensor in self.tensors if isinstance(tensor.op, tvm.tensor.PlaceholderOp)]
    
    @property
    def stages_on_ub(self) -> Set[tvm.schedule.Stage]:
        return set(self.stages) - self.stages_not_on_ub

    @property
    def postorder(self) -> List[tvm.schedule.Stage]:
        def postorder_traversal(root):
            if root in visited:
                return
            visited.add(root)
            for p in self.producer(root):
                postorder_traversal(p)
            postorder.append(root)

        postorder = []
        visited = set()
        postorder_traversal(self.stage_map[self.real_outputs[0]])
        return postorder

    @property
    def reduce_stages(self) -> List[tvm.schedule.Stage]:
        return list(filter(lambda stage: stage.op.tag.find("reduce") != -1, self.stages))

    @property
    def reduce_tensors(self) -> List[tvm.tensor.Tensor]:
        return list(filter(lambda tensor: tensor.op.tag.find("reduce") != -1, self.tensors))

    @property
    def broadcast_stages(self) -> List[tvm.schedule.Stage]:
        return list(filter(lambda stage: stage.op.tag.find("broadcast") != -1, self.stages))
    
    @property
    def broadcast_tensors(self) -> List[tvm.tensor.Tensor]:
        return list(filter(lambda tensor: tensor.op.tag.find("broadcast") != -1, self.tensors))

    @property
    def broadcast_branch(self) -> Set[tvm.schedule.Stage]:
        return set().union(*[self.poset(stage) for stage in self.broadcast_stages])

    @property
    def broadcast_branch_roots(self) -> Set[tvm.schedule.Stage]:
        return set(filter(lambda stage: stage not in self.broadcast_branch, self.broadcast_stages))

    @staticmethod
    def get_insn(stage: tvm.schedule.Stage):
        pass
    
    @staticmethod
    def create_schedule(outs: List[tvm.tensor.Tensor]) -> tvm.schedule.Schedule:
        ops = [t.op for t in outs]
        return tvm.create_schedule(ops)
    
    @staticmethod
    def data_parallel_iteration(stage: tvm.schedule.Stage) -> List[tvm.schedule.IterVar]:
        return [iter_var for iter_var in stage.leaf_iter_vars if iter_var.iter_type == IterType.kDataPar.value]
    
    @staticmethod
    def comm_reduce(stage: tvm.schedule.Stage) -> List[tvm.schedule.IterVar]:
        return [iter_var for iter_var in stage.leaf_iter_vars if iter_var.iter_type == IterType.kCommReduce.value]
    
    @staticmethod
    def reduce_emit_axis(stage: tvm.schedule.Stage) -> tvm.schedule.IterVar:
        for iter_var in stage.leaf_iter_vars:
            if iter_var.iter_type == IterType.kCommReduce.value:
                return iter_var
        return iter_var

    def poset(self, stage: tvm.schedule.Stage) -> Set[tvm.schedule.Stage]:
        """
        partially ordered set
        @param stage:
        @return:
        """
        tensors = set(stage.op.input_tensors)
        queue = set(stage.op.input_tensors)
        while queue:
            tensor = queue.pop()
            tensors.update(tensor.op.input_tensors)
            queue.update(tensor.op.input_tensors)
        stages = set()
        for tensor in tensors:
            stages.add(self.get_stage(tensor))
        return stages
    
    def producer(self, stage: tvm.schedule.Stage) -> List[tvm.schedule.Stage]:
        all_producers = [self.get_stage(tensor) for tensor in stage.op.input_tensors]
        producers = []
        for p in all_producers:
            if p not in producers:
                producers.append(p)
        return producers
    
    def consumer(self, stage: tvm.schedule.Stage) -> List[tvm.schedule.Stage]:
        all_consumers = [_stage for _stage in self.stages if stage in self.producer(_stage)]
        consumers = set()
        for c in all_consumers:
            consumers.add(c)
        return list(consumers)
    
    def get_tensor(self, stage: tvm.schedule.Stage) -> tvm.tensor.Tensor:
        for tensor in self.tensors:
            if tensor.op == stage.op:
                return tensor
        for tensor in self.ori_tensors:
            if tensor.op == stage.origin_op:
                return tensor
        return tensor
    
    def get_ori_tensor(self, stage: tvm.schedule.Stage) -> tvm.tensor.Tensor:
        for tensor in self.ori_tensors:
            if tensor.op == stage.origin_op:
                return tensor
        for tensor in self.tensors:
            if tensor.op == stage.op:
                return tensor
        return tensor
    
    def get_stage(self, tensor: tvm.tensor.Tensor) -> tvm.schedule.Stage:
        for stage in self.stages:
            if tensor.op == stage.op:
                return stage
            if tensor.op == stage.origin_op:
                return stage
        return stage
    
    def cache_read(self, tensor, scope, readers) -> tvm.tensor.Tensor:
        self.stages_not_on_ub.add(self.get_stage(tensor))
        ub_tensor: tvm.tensor.Tensor = self.sch.cache_read(tensor, scope, readers)
        self.cache_read_stages.update({self.get_stage(ub_tensor): tensor})
        self.ori_tensors.append(ub_tensor)
        return ub_tensor
    
    def cache_write(self, tensor, scope) -> tvm.tensor.Tensor:
        for t in tensor:
            self.stages_not_on_ub.add(self.get_stage(t))
        ub_tensor: tvm.tensor.Tensor = self.sch.cache_write(tensor, scope)
        for t in ub_tensor:
            self.cache_write_stages.update({self.get_stage(t): tensor})
            self.ori_tensors.append(t)
        return ub_tensor
    
    def rfactor(self, tensor, axis, factor_axis=0):
        tfactor = self.sch.rfactor(tensor, axis, factor_axis)
        return tfactor

    def info(self):
        pass

    def ir(self):
        pass


class Compute:
    """
    Compute Graph helper
    """

    def __init__(self, outs: List[tvm.tensor.Tensor] or Tuple[tvm.tensor.Tensor]):
        self.ratio = 2
        self.outs: List[tvm.tensor.Tensor] = outs
        self.tensors: List[tvm.tensor.Tensor] = self.postorder_traversal
        self.broadcast_tensors: List[tvm.tensor.Tensor] = list(
            filter(lambda tensor: tensor.op.tag.find("broadcast") != -1, self.tensors))
        self.reduce_tensors: List[tvm.tensor.Tensor] = list(
            filter(lambda tensor: tensor.op.tag.find("reduce") != -1, self.tensors))
        self.broadcast_branch: Set[tvm.tensor.Tensor] = set().union(
            *[self.poset(tensor) for tensor in self.broadcast_tensors])
        self.placeholder: List[tvm.tensor.Tensor] = [
            tensor for tensor in self.tensors if isinstance(tensor.op, tvm.tensor.PlaceholderOp)]

    @property
    def postorder_traversal(self):
        """
        postorder_traversal
        :return:
        """

        def traversal(root):
            if root in visited:
                return
            visited.add(root)
            for p in self.producer(root):
                traversal(p)
            postorder.append(root)

        postorder = []
        visited = set()
        traversal(self.outs[0])
        return postorder
    
    @staticmethod
    def producer(tensor: tvm.tensor.Tensor):
        """
        producers of tensor
        :param tensor:
        :return:
        """
        return tensor.op.input_tensors

    @staticmethod
    def poset(tensor: tvm.tensor.Tensor):
        """
        partially ordered set
        :param tensor:
        :return:
        """
        tensors = set(tensor.op.input_tensors)
        queue = set(tensor.op.input_tensors)
        while queue:
            t = queue.pop()
            tensors.update(t.op.input_tensors)
            queue.update(t.op.input_tensors)
        return tensors

    def consumer(self, tensor: tvm.tensor.Tensor):
        return list(filter(lambda t: tensor in t.op.input_tensors, self.tensors))

    def branch_root(self, subgraph: Iterable[tvm.tensor.Tensor]):
        """
        find the roots of given subgraph
        :param subgraph:
        :return:
        """
        return list(filter(lambda tensor: set(tensor.op.input_tensors).intersection(subgraph)
                                          and tensor not in subgraph, self.tensors))

    def buffer_count(self,
                     grande: Optional[tvm.tensor.Tensor] = None,
                     short: Optional[tvm.tensor.Tensor] = None,
                     unique: Optional[tvm.tensor.Tensor] = None):
        # para_check
        if short is None and grande is None:
            grande = set(self.tensors)
            short = set()
        elif grande is None:
            grande = set(self.tensors) - set(short)
        elif short is None:
            short = set(self.tensors) - set(grande)
        elif set(self.tensors) != set(grande).union(short):
            _raise_error("grande tensor set UNION short tensor set should equal to all tensors in this compute.")

        # process unique
        if unique is None:
            unique = set()

        grande_count = lambda m: sum(key in grande for key in m.keys())
        short_count = lambda m: sum(key in short for key in m.keys())

        allocate_order = self.postorder_traversal
        tensor_alive = {}
        grande_buffer_count = 0
        short_buffer_count = 0

        for tensor in allocate_order:
            if tensor in grande:
                grande_buffer_count = max(grande_count(tensor_alive) + 1, grande_buffer_count)
            if tensor in short:
                if tensor.op.tag.find("tuple_reduce_sum") != -1:
                    short_buffer_count = max(short_count(tensor_alive) + 2, short_buffer_count)
                else:
                    short_buffer_count = max(short_count(tensor_alive) + 1, short_buffer_count)
            tensor_alive.update({tensor: self.consumer(tensor)})
            for k, v in tensor_alive.items():
                tensor_alive[k] = list(filter(lambda x: x != tensor, v))
            tensor_alive = dict(filter(lambda x: len(x[1]) > 0 or x[0] in unique, tensor_alive.items()))

        return [grande_buffer_count, short_buffer_count]

