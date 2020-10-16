/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2019-2020. All rights reserved.
 *
 * @flie   batch_to_space_fusion_pass.cpp
 *
 * @brief  BatchToSpace fusion pass(BatchToSpace --> BatchToSpaceD)
 *
 */

#include "batch_to_space_fusion_pass.h"

#include <iostream>
#include <vector>
#include <map>

#include "graph/utils/op_desc_utils.h"
#include "graph/utils/graph_utils.h"
#include "graph/utils/node_utils.h"
#include "op_log.h"

#include "graph_optimizer/graph_fusion/fusion_pass_manager/fusion_pass_registry.h"
#include "pattern_fusion_util.h"

using namespace std;
using namespace ge;

namespace fe {
static const string PATTERN_BATCH = "BatchToSpace";

vector<FusionPattern*> BatchToSpaceFusionPass::DefinePatterns() {
  vector<FusionPattern*> patterns;

  FusionPattern *pattern = new (std::nothrow) FusionPattern("BatchToSpaceFusion");
  FUSION_PASS_CHECK(pattern == nullptr, OP_LOGE(FUSED_OP_TYPE.c_str(), "new a pattern object failed."),
           return patterns);

  pattern->AddOpDesc(PATTERN_BATCH, {"BatchToSpace"})
      .SetOutput(PATTERN_BATCH);

  patterns.push_back(pattern);

  return patterns;
}

// vector<ge::NodePtr> &fusionNodes: Store fusion nodes,
//       including newly added nodes and fused but not deleted nodes
Status BatchToSpaceFusionPass::Fusion(ge::ComputeGraph &graph,
                                      Mapping &mapping,
                                      vector<ge::NodePtr> &fusionNodes)
{
  ge::NodePtr batchNode = GetNodeFromMapping(PATTERN_BATCH, mapping);
  FUSION_PASS_CHECK(batchNode == nullptr, OP_LOGE(FUSED_OP_TYPE.c_str(), "batchNode is null, fusion failed."),
           return PARAM_INVALID);

  std::vector<PassAttrInfo> attr_infos = {
      {1, "crops", "SetListInt"}
  };
  const std::string fusionOpType = "BatchToSpaceD";
  ge::OpDescPtr fusionDescPtr =
      PatternFusionUtil::GetFusionOpDesc(batchNode, fusionOpType, attr_infos);
  FUSION_PASS_CHECK(fusionDescPtr == nullptr,
           OP_LOGE(FUSED_OP_TYPE.c_str(), "Fusion OP Desc is nullptr."),
           return PARAM_INVALID);
  FUSION_PASS_CHECK(!CheckOpSupported(fusionDescPtr), OP_LOGI(FUSED_OP_TYPE.c_str(), "Op Not Supported."),
           return NOT_CHANGED);

  ge::OpDescPtr batchDesc = batchNode->GetOpDesc();
  FUSION_PASS_CHECK(batchDesc == nullptr,
           OP_LOGE(FUSED_OP_TYPE.c_str(), "batchNode's OpDesc is null, fusion failed."),
           return PARAM_INVALID);
  vector<int64_t> dims = batchDesc->GetOutputDesc("y").GetShape().GetDims();
  for (int64_t ele : dims) {
    if (ele == UNKNOWN_DIM) {
      OP_LOGI(FUSED_OP_TYPE.c_str(), "BatchToSpaceFusionPass got unknown shape, not changed");
      return NOT_CHANGED;
    }
  }

  // 第一个输入为非const的value，第二个输入为const的crops
  ge::InDataAnchorPtr spaceVAnchorPtr2 = batchNode->GetInDataAnchor(1);
  ge::OutDataAnchorPtr constAnchorPtr2 = spaceVAnchorPtr2->GetPeerOutAnchor();
  ge::NodePtr constNode2 = constAnchorPtr2->GetOwnerNode();

  // 设置节点的属性(crops)
  ge::ConstGeTensorPtr constTensor2 = nullptr;
  ge::AttrUtils::GetTensor(constNode2->GetOpDesc(), "value", constTensor2);
  size_t constSize2 = constTensor2->GetData().GetSize();
  const uint8_t* constData2 = constTensor2->GetData().GetData();
  ge::DataType constType2 = constTensor2->GetTensorDesc().GetDataType();

  size_t numsize2;
  if (constType2 == ge::DT_INT32) {
    numsize2 = constSize2 / sizeof(int32_t);
    vector<int32_t> crops;
    for (size_t i = 0; i < numsize2; i++) {
      crops.push_back(*((int32_t*)constData2 + i));
    }
    ge::AttrUtils::SetListInt(batchDesc, "crops", crops);
  } else {
    numsize2 = constSize2 / sizeof(int64_t);
    vector<int64_t> crops;
    for (size_t i = 0; i < numsize2; i++) {
      crops.push_back(*((int64_t*)constData2 + i));
    }
    ge::AttrUtils::SetListInt(batchDesc, "crops", crops);
  }

  // 删除const节点、输入节点和边
  ge::GraphUtils::RemoveEdge(constAnchorPtr2, spaceVAnchorPtr2);
  ge::NodeUtils::ClearInDataAnchor(batchNode, spaceVAnchorPtr2);
  ge::OpDescUtils::ClearInputDesc(batchDesc, 1);
  if (PatternFusionUtil::GetOutEdgeSize(constNode2) == 0) {
    FUSION_PASS_CHECK(ge::GRAPH_SUCCESS != graph.RemoveNode(constNode2),
    OP_LOGE(FUSED_OP_TYPE.c_str(), "Remove Node[%s] failed", constNode2->GetName().c_str()),
            return FAILED);
    OP_LOGI(FUSED_OP_TYPE.c_str(), "Remove const Node:[%s].", constNode2->GetName().c_str());
  }
  vector<bool> is_input_const = {false};
  batchDesc->SetIsInputConst(is_input_const);

  // 设置算子type
  batchDesc->SetType(fusionOpType);
  fusionNodes.push_back(batchNode);

  return SUCCESS;
}

REGISTER_PASS("BatchToSpace", BUILT_IN_GRAPH_PASS, BatchToSpaceFusionPass);
}
