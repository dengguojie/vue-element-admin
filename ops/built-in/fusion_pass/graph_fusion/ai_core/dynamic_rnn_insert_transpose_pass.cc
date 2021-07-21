/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2019-2019. All rights reserved.
 *
 * @brief decode_bbox_v2 fusion pass
 *
 */
#include "dynamic_rnn_insert_transpose_pass.h"
#include <fstream>
#include <iostream>
#include <map>
#include <memory>
#include <sstream>
#include <vector>
#include <algorithm>
#include "op_log.h"
#include "error_util.h"
#include "graph/debug/ge_attr_define.h"
#include "graph/utils/attr_utils.h"
#include "graph/utils/graph_utils.h"
#include "graph/utils/node_utils.h"
#include "graph/utils/op_desc_utils.h"
#include "graph_optimizer/graph_fusion/fusion_pass_manager/fusion_pass_registry.h"
#include "pattern_fusion_util.h"
#include "tbe_fusion_pass_util.h"

namespace fe {
static const string PATTERN_FUSEDNODE = "FusedNodeDynamicRNN";
static const string FUSED_NODE = "DynamicRNN";
static const string FUSED_NODE_V2 = "DynamicRNNV2";

vector<FusionPattern*> DynamicRNNInsertTransposePass::DefinePatterns() {
  vector<FusionPattern*> patterns;
  FusionPattern* pattern = new (std::nothrow) FusionPattern("DynamicRNNInsertTransposePass");
  FUSION_PASS_CHECK(pattern == nullptr, VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "New a pattern object failed."),
                    return patterns);
  pattern->AddOpDesc(PATTERN_FUSEDNODE, {FUSED_NODE, FUSED_NODE_V2}).SetOutput(PATTERN_FUSEDNODE);
  patterns.push_back(pattern);
  return patterns;
}

Status DynamicRNNInsertTransposePass::Fusion(ge::ComputeGraph& graph, Mapping& mapping, vector<ge::NodePtr>& newNodes) {
  OP_LOGI(FUSED_OP_TYPE.c_str(), "Define DynamicRNNInsertTransposePass fusion begin.");
  ge::NodePtr fusedNode = GetNodeFromMapping(PATTERN_FUSEDNODE, mapping);
  FUSION_PASS_CHECK(fusedNode == nullptr, VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "fusedNode is null, fusion failed."),
                    return PARAM_INVALID);
  ge::OpDescPtr fuseDesc = fusedNode->GetOpDesc();
  bool time_major = true;
  FUSION_PASS_CHECK(!ge::AttrUtils::GetBool(fuseDesc, "time_major", time_major),
                    VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "get DynamicRnn's attr time_major failed."), return FAILED);
  if (!time_major) {
    // do infer for fused node again, and update fused node output shape
    ge::GeTensorDesc outputDesc = fusedNode->GetOpDesc()->GetOutputDesc(0);
    vector<int64_t> oriOutputShape = outputDesc.GetShape().GetDims();

    if (oriOutputShape.size() < 3) {
      OP_LOGW(FUSED_OP_TYPE.c_str(), "can not get output shape. shape less then 3!");
      return NOT_CHANGED;
    }

    vector<int64_t> permBoxesList = {1, 0, 2};
    AddTransposeBeforeNode(fusedNode, 0, permBoxesList, graph);

    vector<int64_t> outputShapeVec;
    outputShapeVec.push_back(oriOutputShape[1]);
    outputShapeVec.push_back(oriOutputShape[0]);
    outputShapeVec.push_back(oriOutputShape[2]);
    ge::GeShape outputShape(outputShapeVec);
    outputDesc.SetShape(outputShape);
    outputDesc.SetOriginShape(outputShape);
    // update fused node output info
    auto opOutputDesc = fusedNode->GetOpDesc();
    opOutputDesc->UpdateOutputDesc(0, outputDesc);
    opOutputDesc->UpdateOutputDesc(1, outputDesc);
    opOutputDesc->UpdateOutputDesc(2, outputDesc);
    opOutputDesc->UpdateOutputDesc(3, outputDesc);
    opOutputDesc->UpdateOutputDesc(4, outputDesc);
    opOutputDesc->UpdateOutputDesc(5, outputDesc);
    opOutputDesc->UpdateOutputDesc(6, outputDesc);
    opOutputDesc->UpdateOutputDesc(7, outputDesc);
    AddTransposeAfterNode(fusedNode, 0, permBoxesList, graph);
    FUSION_PASS_CHECK(!ge::AttrUtils::SetBool(fuseDesc, "time_major", true),
                      VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "Set DynamicRnn's attr time_major true failed."), return FAILED);
  } else {
    OP_LOGI(FUSED_OP_TYPE.c_str(), "DynamicRnn time_major is true don't need insert Transpose");
  }
  OP_LOGI(FUSED_OP_TYPE.c_str(), "Define DynamicRNNInsertTransposePass fusion end");

  return SUCCESS;
}

REGISTER_PASS("DynamicRNNInsertTransposePass", BUILT_IN_GRAPH_PASS, DynamicRNNInsertTransposePass);
}  // namespace fe
