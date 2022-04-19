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

#ifndef FUSION_ENGINE_STUB_FUSIONPASSSLICEUTILS_H
#define FUSION_ENGINE_STUB_FUSIONPASSSLICEUTILS_H

#include <string>
#include <vector>

#include "common/lxfusion_json_util.h"
#include "graph/compute_graph.h"
#include "graph/graph.h"
#include "graph/utils/attr_utils.h"
#include "graph_optimizer/fusion_common/op_slice_info.h"
#include "pattern_fusion_util.h"

namespace fe {
void SetSplitMapMainNode(std::vector<AxisSplitMap> &split_maps, std::vector<ge::NodePtr> &Nodes, const string &op_type);

bool SetSplitMapToNodeByType(const ge::ComputeGraphPtr compute_graph_ptr, std::vector<AxisSplitMap> &vec_split_map,
                             const std::vector<string> &type_ops);
bool SetSplitMapToNodeByName(const ge::ComputeGraphPtr compute_graph_ptr, std::vector<AxisSplitMap> &vec_split_map,
                             const string &namekk_op);

string GetFusionOpSliceInfoStrFromGraph(const ge::ComputeGraphPtr compute_graph_ptr);

string CreateFusionOpSliceInfoStrFromSplitMap(const std::vector<fe::AxisSplitMap> &vec_split_map);

InputSplitInfo CreateInputSplitInfo(const size_t &idx, const std::vector<int64_t> &axis,
                                    const std::vector<int64_t> &head_over_lap = {-1},
                                    const std::vector<int64_t> &tail_over_lap = {-1});

OutputSplitInfo CreateOutputSplitInfo(const size_t &idx, const std::vector<int64_t> &axis);

AxisSplitMap CreateAxisSplitMap(const std::vector<InputSplitInfo> &vec_input_split_info,
                                const std::vector<OutputSplitInfo> &vec_output_split_info);
}  // namespace fe

#endif  // FUSION_ENGINE_STUB_FUSIONPASSSLICEUTILS_H
