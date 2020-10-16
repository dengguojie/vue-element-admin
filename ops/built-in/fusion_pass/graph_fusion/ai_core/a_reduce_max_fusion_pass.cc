/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2019-2019. All rights reserved.
 *
 * @brief reducemax fusion pass
 *
 */

#include "a_reduce_max_fusion_pass.h"
#include <fstream>
#include <iostream>
#include <map>
#include <memory>
#include <sstream>
#include <vector>
#include <algorithm>
#include "op_log.h"
#include "graph/debug/ge_attr_define.h"
#include "graph/utils/attr_utils.h"
#include "graph/utils/graph_utils.h"
#include "graph/utils/node_utils.h"
#include "graph/utils/op_desc_utils.h"
#include "graph_optimizer/graph_fusion/fusion_pass_manager/fusion_pass_registry.h"
#include "pattern_fusion_util.h"

namespace fe {
static const string PATTERN_FUSEDNODE = "FusedNodeReduceMax";
static const string FUSED_NODE = "ReduceMax";

Status AReduceMaxFusionPass::CheckMaxFussionOrNot(vector<int64_t> tensor_info,
  vector<int64_t> axis_info) {
  for (auto &dim : tensor_info) {
    if (dim < 0) {
      OP_LOGI(FUSED_OP_TYPE.c_str(), "Dynamic shape process, shouldn't delete.");
      return FAILED;
    }
  }
  for (size_t i = 0; i < axis_info.size(); ++i) {
    if (tensor_info[axis_info[i]] != 1) {
      return FAILED;
    }
  }
  return SUCCESS;
}

vector<FusionPattern *> AReduceMaxFusionPass::DefinePatterns() {
  vector<FusionPattern *> patterns;
  FusionPattern *pattern = \
  new (std::nothrow) FusionPattern("AReduceMaxFusionPass");
  FUSION_PASS_CHECK(pattern == nullptr, OP_LOGE(FUSED_OP_TYPE.c_str(), "New a pattern object failed."),  return patterns);
  pattern->AddOpDesc(PATTERN_FUSEDNODE, {FUSED_NODE})
          .SetOutput(PATTERN_FUSEDNODE);
  patterns.push_back(pattern);
  return patterns;
}

Status AReduceMaxFusionPass::Fusion(ge::ComputeGraph &graph,
    Mapping &mapping, vector<ge::NodePtr> &newNodes) {
  OP_LOGI(FUSED_OP_TYPE.c_str(), "Define AReduceMaxFusionPass fusion begin.");
  ge::NodePtr maxNode = \
  GetNodeFromMapping(PATTERN_FUSEDNODE, mapping);
  FUSION_PASS_CHECK(maxNode == nullptr,
           OP_LOGE(FUSED_OP_TYPE.c_str(), "maxNode is null, fusion failed."),
           return PARAM_INVALID);

  ge::GeTensorDesc tensor_input = \
  maxNode->GetOpDesc()->GetInputDesc(0);
  ge::GeTensorDesc axis_input = \
  maxNode->GetOpDesc()->GetInputDesc(1);

  vector<int64_t> tensor_info = tensor_input.GetShape().GetDims();
  size_t tensor_size = tensor_input.GetShape().GetDimNum();

  vector<int64_t> axis_info = axis_input.GetShape().GetDims();

  Operator op = ge::OpDescUtils::CreateOperatorFromNode(maxNode);
  Tensor data;
  if (GRAPH_SUCCESS != op.GetInputConstData("axes", data)) {
    OP_LOGI(FUSED_OP_TYPE.c_str(), "GetInputConstData of axes failed.");
    return NOT_CHANGED;
  }

  std::vector<int64_t> const_data;
  int32_t* const_data_ptr = (int32_t*) data.GetData();
  size_t const_data_size = data.GetSize() / sizeof(int32_t);
  for (size_t i = 0; i < const_data_size; ++i) {
      const_data.push_back((int32_t) ((*(const_data_ptr + i))));
  }

  if (const_data_size == 0) {
    for (size_t i = 0; i < tensor_info.size(); ++i) {
      const_data.push_back(i);
    }
  }

  for (size_t i = 0; i < const_data_size; ++i) {
    if (const_data[i] < 0) {
      const_data[i] = tensor_size + const_data[i];
    }
  }

  if (!(CheckMaxFussionOrNot(tensor_info, const_data) == SUCCESS)) {
    OP_LOGI(FUSED_OP_TYPE.c_str(), "Not need delete maxNode");
    return NOT_CHANGED;
  }

  OP_LOGI(FUSED_OP_TYPE.c_str(), "delete edge of afterNode and max. connect beforeNode and afterNode");
  for (auto inDataAnchor :
       maxNode->GetOutDataAnchor(0)->GetPeerInDataAnchors()) {
    FUSION_PASS_CHECK(ge::GraphUtils::RemoveEdge(maxNode->GetOutDataAnchor(0),
                                        inDataAnchor) != SUCCESS,
             OP_LOGE(FUSED_OP_TYPE.c_str(), "Remove max and outnode edge failed."), return FAILED);
    FUSION_PASS_CHECK(ge::GraphUtils::AddEdge(maxNode->GetInDataAnchor(0)->GetPeerOutAnchor(),
                                     inDataAnchor) != SUCCESS,
             OP_LOGE(FUSED_OP_TYPE.c_str(), "Add innode and outnode edge failed."), return FAILED);
  }

  OP_LOGI(FUSED_OP_TYPE.c_str(), "delete reducemax edge.");
  FUSION_PASS_CHECK(graph.RemoveNode(maxNode) != SUCCESS,
           OP_LOGE(FUSED_OP_TYPE.c_str(), "Remove maxNode failed."), return FAILED);

  OP_LOGI(FUSED_OP_TYPE.c_str(), "Define AReduceMaxFusionPass fusion end");

  return SUCCESS;
  }

REGISTER_PASS("AReduceMaxFusionPass", BUILT_IN_GRAPH_PASS,
              AReduceMaxFusionPass);
}
