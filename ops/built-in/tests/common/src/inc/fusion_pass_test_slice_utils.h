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
#include "graph/graph.h"
#include "graph/compute_graph.h"
#include "pattern_fusion_util.h"
#include "graph_optimizer/fusion_common/op_slice_info.h"
#include "graph/utils/attr_utils.h"
#include "common/lxfusion_json_util.h"

namespace fe {

    static const std::string OP_SLICE_INFO = "_op_slice_info";

    void SetSplitMapMainNode(std::vector<AxisSplitMap>& split_maps,
                             std::vector<ge::NodePtr>& Nodes, const string& op_type);

}  // namespace fe

#endif  // FUSION_ENGINE_STUB_FUSIONPASSSLICEUTILS_H
