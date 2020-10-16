/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2019-2019. All rights reserved.
 *
 * @brief reducesum fusion pass
 *
 */

#include "a_reduce_sum_fusion_pass.h"
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
static const string PATTERN_FUSEDNODE = "FusedNodeReduceSum";
static const string FUSED_NODE = "ReduceSum";

Status AReduceSumFusionPass::CheckSumFussionOrNot(vector<int64_t> tensor_info,
  vector<int64_t> axis_info) {
  for (auto &input_shape_value : tensor_info) {
    if (input_shape_value < 0) {
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

vector<FusionPattern *> AReduceSumFusionPass::DefinePatterns() {
  vector<FusionPattern *> patterns;
  FusionPattern *pattern = \
  new (std::nothrow) FusionPattern("AReduceSumFusionPass");
  FUSION_PASS_CHECK(pattern == nullptr, OP_LOGE(FUSED_OP_TYPE.c_str(), "New a pattern object failed."),  return patterns);
  pattern->AddOpDesc(PATTERN_FUSEDNODE, {FUSED_NODE})
          .SetOutput(PATTERN_FUSEDNODE);
  patterns.push_back(pattern);
  return patterns;
}

Status AReduceSumFusionPass::Fusion(ge::ComputeGraph &graph,
    Mapping &mapping, vector<ge::NodePtr> &newNodes) {
  OP_LOGI(FUSED_OP_TYPE.c_str(), "Define AReduceSumFusionPass fusion begin.");
  ge::NodePtr sumNode = \
  GetNodeFromMapping(PATTERN_FUSEDNODE, mapping);
  FUSION_PASS_CHECK(sumNode == nullptr,
           OP_LOGE(FUSED_OP_TYPE.c_str(), "sumNode is null, fusion failed."),
           return PARAM_INVALID);

  ge::GeTensorDesc tensor_input = \
  sumNode->GetOpDesc()->GetInputDesc(0);
  ge::GeTensorDesc axis_input = \
  sumNode->GetOpDesc()->GetInputDesc(1);

  vector<int64_t> tensor_info = tensor_input.GetShape().GetDims();
  size_t tensor_size = tensor_input.GetShape().GetDimNum();

  vector<int64_t> axis_info = axis_input.GetShape().GetDims();
  size_t axis_size = axis_input.GetShape().GetDimNum();

  Operator op = ge::OpDescUtils::CreateOperatorFromNode(sumNode);
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

  int axis_value = axis_input.GetShape().GetDim(0);

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

  if (!(CheckSumFussionOrNot(tensor_info, const_data) == SUCCESS) && (axis_size!=1 || axis_value!=0)) {
    OP_LOGI(FUSED_OP_TYPE.c_str(), "Not need delete sumNode");
    return NOT_CHANGED;
  }

  OP_LOGI(FUSED_OP_TYPE.c_str(), "delete edge of afterNode and sum. connect beforeNode and afterNode");
  for (auto inDataAnchor :
       sumNode->GetOutDataAnchor(0)->GetPeerInDataAnchors()) {
    FUSION_PASS_CHECK(ge::GraphUtils::RemoveEdge(sumNode->GetOutDataAnchor(0),
                                        inDataAnchor) != SUCCESS,
             OP_LOGE(FUSED_OP_TYPE.c_str(), "Remove sum and outnode edge failed."), return FAILED);
    FUSION_PASS_CHECK(ge::GraphUtils::AddEdge(sumNode->GetInDataAnchor(0)->GetPeerOutAnchor(),
                                     inDataAnchor) != SUCCESS,
             OP_LOGE(FUSED_OP_TYPE.c_str(), "Add innode and outnode edge failed."), return FAILED);
  }

  OP_LOGI(FUSED_OP_TYPE.c_str(), "delete reducesum edge.");
  FUSION_PASS_CHECK(graph.RemoveNode(sumNode) != SUCCESS,
           OP_LOGE(FUSED_OP_TYPE.c_str(), "Remove sumNode failed."), return FAILED);

  OP_LOGI(FUSED_OP_TYPE.c_str(), "Define AReduceSumFusionPass fusion end");

  return SUCCESS;
  }

REGISTER_PASS("AReduceSumFusionPass", BUILT_IN_GRAPH_PASS,
              AReduceSumFusionPass);
}
