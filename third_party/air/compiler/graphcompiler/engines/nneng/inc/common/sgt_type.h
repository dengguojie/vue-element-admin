/**
 * Copyright 2019-2020 Huawei Technologies Co., Ltd
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#ifndef FUSION_ENGINE_INC_COMMON_SGT_TYPES_H_
#define FUSION_ENGINE_INC_COMMON_SGT_TYPES_H_

#include <map>
#include <string>
#include <vector>
#include "graph/anchor.h"
#include "graph/types.h"

using std::vector;
using std::pair;
using std::string;
using std::map;

namespace fe {
const std::string SGT_JSON_INFO = "_sgt_json_info";
const std::string SGT_STRUCT_INFO = "_sgt_struct_info";


struct DimRange {
  int64_t lower;
  int64_t higher;
  bool operator==(const DimRange& dim_range) const {
    return this->higher == dim_range.higher &&
           this->lower == dim_range.lower;
  }
};

struct ThreadSliceMap {
  uint32_t thread_scopeId;
  bool is_first_node_in_topo_order;
  uint32_t node_num_in_thread_scope;
  uint32_t slice_instance_num;
  uint32_t parallel_window_size;
  vector<uint32_t> thread_id;
  vector<vector<pair<string, uint32_t>>> dependencies;
  vector<uint32_t> core_num;
  vector<vector<vector<DimRange>>> input_tensor_slice;
  vector<vector<vector<DimRange>>> output_tensor_slice;
  vector<vector<vector<DimRange>>> ori_input_tensor_slice;
  vector<vector<vector<DimRange>>> ori_output_tensor_slice;
  ThreadSliceMap() {
    is_first_node_in_topo_order = false;
  }
};

struct TickCacheMap {
  vector<int32_t> src_out_of_graph_input_index;
  map<int32_t, uint8_t> input_cache_table;
  map<int32_t, uint8_t> output_cache_table;
};

using ThreadSliceMapPtr = std::shared_ptr<ThreadSliceMap>;
}
#endif  // FUSION_ENGINE_INC_COMMON_SGT_TYPES_H_
