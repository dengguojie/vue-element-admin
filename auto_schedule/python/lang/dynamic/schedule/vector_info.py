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
Information containers for compute graph of vector operator schedule
"""

# Standard Packages
from typing import Any
from typing import Set
from typing import Dict
from typing import List
from typing import Tuple
from typing import Union
from typing import Optional
from typing import Callable
from typing import Iterable

# Third-party packages
from te.tvm.tensor import Tensor
from te.tvm.tensor import PlaceholderOp
from te.platform.cce_conf import get_soc_spec


class ComputeGraphInfo:
    """
    Operator Compute Graph Info collector and container
    """
    def __init__(self, output_tensors: Iterable[Tensor]):
        """
        Initialize containers and try to collect info
        """
        # Basic Info
        self.output_tensor_set: Optional[Set[Tensor]] = None
        self.tensor_consumers_map: Optional[Dict[Tensor, Set[Tensor]]] = None
        self.tensor_producers_map: Optional[Dict[Tensor, Set[Tensor]]] = None
        self.tensor_list: Optional[List[Tensor]] = None
        # Extra info initialized by hooks
        self.reduce_tensor_set: Set[Tensor] = set()
        self.broadcast_tensor_set: Set[Tensor] = set()
        self.elewise_tensor_set: Set[Tensor] = set()
        self.input_tensor_set: Set[Tensor] = set()
        self.non_gm_input_tensor_set: Set[Tensor] = set()
        # Extra info initialized after pre-initialization
        self.mid_output_tensor_set: Set[Tensor] = set()
        self.mid_tensor_set: Set[Tensor] = set()
        self.endpoint_output_tensor_set: Set[Tensor] = set()
        self.max_single_tensor_ub_size: Optional[int] = None
        # Do info collection
        self._collect_info(output_tensors)
        self._init_max_ub_count()

    def _collect_info(self, output_tensors: Iterable[Tensor]):
        self.output_tensor_set = set(output_tensors)
        self.tensor_list, self.tensor_consumers_map, self.tensor_producers_map = \
            self.dfs_compute_graph(self.output_tensor_set,
                                   (   # self.input_tensor_set hook
                                       (lambda _tensor: isinstance(_tensor.op, PlaceholderOp),
                                        lambda _tensor: self.input_tensor_set.add(_tensor),
                                        lambda _tensor: self.non_gm_input_tensor_set.add(_tensor)
                                        if not _tensor.op.input_tensors else None),
                                       # self.reduce_tensor_set hook
                                       (lambda _tensor: _tensor.op.tag.find("reduce") != -1,
                                        lambda _tensor: self.reduce_tensor_set.add(_tensor),
                                        lambda _tensor: None),
                                       # self.broadcast_tensor_set hook
                                       (lambda _tensor: _tensor.op.tag.find("broadcast") != -1,
                                        lambda _tensor: self.broadcast_tensor_set.add(_tensor),
                                        lambda _tensor: None),
                                       # self.elewise_tensor_set hook
                                       (lambda _tensor: _tensor.op.tag.find("elewise") != -1,
                                        lambda _tensor: self.elewise_tensor_set.add(_tensor),
                                        lambda _tensor: None)
                                   ))
        # Initialize non-hookable info
        self.gen_mid_tensor_sets()
        # endpoint_output_tensor_set
        self.gen_endpoint_output_tensor_set()

    def gen_endpoint_output_tensor_set(self):
        for output_tensor in self.output_tensor_set:
            if not self.tensor_consumers_map[output_tensor]:
                self.endpoint_output_tensor_set.add(output_tensor)

    def gen_mid_tensor_sets(self):
        # mid_output_tensor_set
        # mid_tensor_set
        for tensor in self.tensor_list:
            if tensor in self.output_tensor_set and self.tensor_consumers_map[tensor]:
                # Tensor in output and has consumers is middle_out_tensor
                self.mid_output_tensor_set.add(tensor)
                self.mid_tensor_set.add(tensor)
            elif tensor not in self.output_tensor_set | self.input_tensor_set | self.non_gm_input_tensor_set:
                self.mid_tensor_set.add(tensor)

    @staticmethod
    def dfs_compute_graph(root_tensor: Union[Iterable[Tensor], Tensor],
                          hooks: Tuple[Tuple[Callable[[Tensor], bool],
                                             Callable[[Tensor], Any],
                                             Callable[[Tensor], Any]], ...]):
        def recursive_func(_root_tensor: Tensor,
                           _visited_list: Set[Tensor],
                           _tensor_consumers_map: Dict[Tensor, Union[Set[Tensor]]],
                           _tensor_producers_map: Dict[Tensor, Union[Set[Tensor]]],
                           _hooks: Tuple[Tuple[Callable[[Tensor], bool],
                                               Callable[[Tensor], Any],
                                               Callable[[Tensor], Any]], ...]):
            _visited_list.add(_root_tensor)
            _tensor_producers_map.setdefault(_root_tensor, set())
            _tensor_consumers_map.setdefault(_root_tensor, set())
            for hook in hooks:
                if hook[0](_root_tensor):
                    hook[1](_root_tensor)
                else:
                    hook[2](_root_tensor)
            for in_tensor in _root_tensor.op.input_tensors:
                _tensor_consumers_map.setdefault(in_tensor, set())
                _tensor_consumers_map[in_tensor].add(_root_tensor)
                _tensor_producers_map[_root_tensor].add(in_tensor)
                recursive_func(in_tensor,
                               _visited_list,
                               _tensor_consumers_map,
                               _tensor_producers_map,
                               _hooks)
        visited_list = set()
        tensor_consumers_map = {}
        tensor_producers_map = {}
        if isinstance(root_tensor, (list, tuple, set)):
            for tensor in root_tensor:
                recursive_func(tensor, visited_list,
                               tensor_consumers_map,
                               tensor_producers_map,
                               hooks)
        elif isinstance(root_tensor, Tensor):
            recursive_func(root_tensor, visited_list,
                           tensor_consumers_map, tensor_producers_map,
                           hooks)
        else:
            raise RuntimeError("dfs_compute_graph() supports list, tuple, Tensor only. Received %s"
                               % str(type(root_tensor)))
        return list(visited_list), tensor_consumers_map, tensor_producers_map

    def _init_max_ub_count(self):
        soc_ub_size = get_soc_spec("UB_SIZE")
        soc_ub_size = soc_ub_size // 2
        total_width = 8
        if not total_width:
            raise RuntimeError("Could not get compute width")
        max_bound = total_width * 128
        max_ub_count = int(soc_ub_size // max_bound * 128)

        self.max_single_tensor_ub_size = max_ub_count

    @staticmethod
    def set_map_deepcopy(_map: Dict[Tensor, Set[Tensor]]) -> Dict[Tensor, Set[Tensor]]:
        return dict((key, _map[key].copy()) for key in _map)
