/* Copyright (c) Huawei Technologies Co., Ltd. 2012-2020. All rights reserved.
 *
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the Apache License Version 2.0.
 * You may not use this file except in compliance with the License.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
 * Apache License for more details at
 * http://www.apache.org/licenses/LICENSE-2.0
 */

#include "globalavgpool_fusion_pass.h"
#include <vector>
#include <memory>
#include "graph/utils/op_desc_utils.h"
#include "graph/utils/graph_utils.h"
#include "graph/utils/node_utils.h"
#include "graph/utils/attr_utils.h"
#include "graph/debug/ge_attr_define.h"
#include "op_log.h"
#include "graph_optimizer/graph_fusion/fusion_pass_manager/fusion_pass_registry.h"
#include "tbe_fusion_pass_util.h"
namespace fe {
  const std::string Globalavgpoolpass::PATTERN_FUSEDNODE = "GlobalAveragePool";
  vector<FusionPattern *> Globalavgpoolpass::DefinePatterns() {
    vector<FusionPattern *> patterns;
    FusionPattern* pattern = (new(std::nothrow)
                              FusionPattern("Globalavgpoolpass"));
    FUSION_PASS_CHECK(pattern == nullptr,
                     OP_LOGE(FUSED_OP_TYPE.c_str(), "New a pattern object failed."),
                     return patterns);
    pattern->AddOpDesc(PATTERN_FUSEDNODE, {"GlobalAveragePool"}).SetOutput(PATTERN_FUSEDNODE);
    patterns.push_back(pattern);
    return patterns;
  }

  Status Globalavgpoolpass::Fusion(ge::ComputeGraph &graph,
                                   Mapping &mapping,
                                   vector<ge::NodePtr>& fusion_nodes) {
    ge::NodePtr fused_node = GetNodeFromMapping(PATTERN_FUSEDNODE, mapping);
    FUSION_PASS_CHECK(fused_node == nullptr,
                      OP_LOGE(FUSED_OP_TYPE.c_str(), "Fusion GetNode Error"),
                      return PARAM_INVALID);
    ge::OpDescPtr fuse_desc = fused_node->GetOpDesc();
    FUSION_PASS_CHECK(fuse_desc == nullptr,
                OP_LOGE(FUSED_OP_TYPE.c_str(), "fuse_node's OpDesc is null, fusion failed."),
                return PARAM_INVALID);

    ge::GeTensorDesc input_x_desc = fused_node->GetOpDesc()->GetInputDesc("x");
    int64_t input_x_dims = input_x_desc.GetShape().GetDims().size();
    fuse_desc->SetType("ReduceMeanD");
    bool keep_f = true;
    std::vector<int> axes_num = {2, 3, 4};
    if (!ge::AttrUtils::SetBool(fuse_desc, "keep_dims", keep_f)) {
      OP_LOGE(FUSED_OP_TYPE.c_str(), "Set attr keep_dims failed.");
      return FAILED;
    }

    if (input_x_dims == 3) {
      axes_num = {2};
    } else if (input_x_dims == 4) {
      axes_num = {2, 3};
    } else if (input_x_dims == 5) {
      axes_num = {2, 3, 4};
    } else {
      OP_LOGE(FUSED_OP_TYPE.c_str(), "input shape failed.");
      return FAILED;
    }
    if (!ge::AttrUtils::SetListInt(fuse_desc, "axes", axes_num)) {
      OP_LOGE(FUSED_OP_TYPE.c_str(), "Set attr axes failed.");
      return FAILED;
    }
    return SUCCESS;
  }
  REGISTER_PASS("Globalavgpoolpass", BUILT_IN_GRAPH_PASS, Globalavgpoolpass);
}