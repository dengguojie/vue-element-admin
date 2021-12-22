/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2019-2021. All rights reserved.
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

/*!
 * \file slice_fusion_pass.cc
 * \brief split fusion pass(slice --> slice_d)
 */
#include "slice_fusion_pass.h"

#include <math.h>

#include <iostream>
#include <map>
#include <string>
#include <vector>

#include "graph/debug/ge_attr_define.h"
#include "graph/utils/attr_utils.h"
#include "graph/utils/graph_utils.h"
#include "graph/utils/node_utils.h"
#include "graph/utils/op_desc_utils.h"
#include "graph_optimizer/graph_fusion/fusion_pass_manager/fusion_pass_registry.h"
#include "op_log.h"
#include "error_util.h"
#include "pattern_fusion_util.h"
#include "tbe_fusion_pass_util.h"
#include "tbe_ops_pass_util.h"

using namespace ge;
namespace fe {

static const char* FUSED_NODE = "Slice";

static const std::string PATTERN_FUSEDNODE = "FusedNodeSlice";

vector<FusionPattern*> ConstToAttrSlicePass::DefinePatterns() {
  vector<FusionPattern*> patterns;

  FusionPattern* pattern = new (std::nothrow) FusionPattern("ConstToAttrSliceFusion");
  FUSION_PASS_CHECK(pattern == nullptr, VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(),
                    "new a pattern object failed."),
                    return patterns);

  pattern->AddOpDesc(PATTERN_FUSEDNODE, {FUSED_NODE}).SetOutput(PATTERN_FUSEDNODE);

  patterns.push_back(pattern);

  return patterns;
}

Status ConstToAttrSlicePass::Fusion(ge::ComputeGraph& graph, Mapping& mapping, vector<ge::NodePtr>& fusionNodes) {
  // PatternFusionUtil patternFusionUtil;
  ge::NodePtr fused_node = GetNodeFromMapping(PATTERN_FUSEDNODE, mapping);
  FUSION_PASS_CHECK(fused_node == nullptr, VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(),
                    "fused_node's Node is null, fusion failed."),
                    return PARAM_INVALID);
  ge::OpDescPtr fuseDesc = fused_node->GetOpDesc();
  FUSION_PASS_CHECK(fuseDesc == nullptr, VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(),
                    "fused_node's OpDesc is null, fusion failed."),
                    return PARAM_INVALID);

  if (HasUnKnowShape(fused_node)) {
    FUSION_PASS_CHECK(CheckOpSupported(fuseDesc), OP_LOGI(FUSED_NODE, "dynamic shape supported."), return NOT_CHANGED);
    OP_LOGD(FUSED_NODE, "CheckOpSupported false.");
  }

  vector<int64_t> dims = fuseDesc->GetOutputDesc(0).GetShape().GetDims();
  for (int64_t ele : dims) {
    if (ele == 0) {
      OP_LOGI(FUSED_OP_TYPE.c_str(), "The output of slice's dim is 0, need go to aicpu");
      return NOT_CHANGED;
    }
  }

  FUSION_PASS_CHECK(fused_node == nullptr, VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(),
                    "fused_node is null, fusion failed."),
                    return PARAM_INVALID);

  Operator op = ge::OpDescUtils::CreateOperatorFromNode(fused_node);
  std::vector<int64_t> ends;
  if (!TbeFusionPassUtil::GetConstIntData(op, "size", ends)) {
    OP_LOGI(FUSED_OP_TYPE.c_str(), "Slice get size value failed.");
    return NOT_CHANGED;
  }
  std::vector<int64_t> begins;
  std::string fusion_op_type;
  std::vector<PassAttrInfo> attr_infos;
  if (!TbeFusionPassUtil::GetConstIntData(op, "offsets", begins)) {
    fusion_op_type = "SliceDV2";
    OP_LOGI(FUSED_OP_TYPE.c_str(), "Slice get begin const failed.");
    attr_infos = {{2, "size", "SetListInt"}};
  } else {
    fusion_op_type = "SliceD";
    attr_infos = {{1, "offsets", "SetListInt"}, {2, "size", "SetListInt"}};
  }

  ge::OpDescPtr desc_ptr = PatternFusionUtil::GetFusionOpDesc(fused_node, fusion_op_type, attr_infos);
  FUSION_PASS_CHECK(desc_ptr == nullptr, OP_LOGI(FUSED_OP_TYPE.c_str(), "The input is not a constant."),
                    return NOT_CHANGED);
  FUSION_PASS_CHECK(!CheckOpSupported(desc_ptr), OP_LOGI(FUSED_OP_TYPE.c_str(), "Op Not Supported."),
                    return NOT_CHANGED);

  ge::NodePtr fusionNode = nullptr;
  auto ret = PatternFusionUtil::ConstToAttrWithNode(graph, fused_node, fusion_op_type, attr_infos, fusionNode);
  if (ret != SUCCESS) {
    OP_LOGI(FUSED_OP_TYPE.c_str(), "Slice has input which is not a CONST, graph not changed.");
    return NOT_CHANGED;
  }
  ClearOpInferDepends(fused_node);
  fusionNodes.push_back(fusionNode);
  return SUCCESS;
}

REGISTER_PASS("ConstToAttrSliceFusion", BUILT_IN_GRAPH_PASS, ConstToAttrSlicePass);
}  // namespace fe