/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2019-2019. All rights reserved.
 *
 * @brief confusionTranspose fusion pass(Transpose-Reshape --> confusionTranspose)
 *
 */
 
// .h和.cpp文件提交至ops/built-in/fusion_pass/目录下

#include "transpose_reshape_fusion_pass.h"

#include <iostream>
#include <vector>
#include <string>
#include <map>

#include "graph/utils/op_desc_utils.h"
#include "graph/utils/graph_utils.h"
#include "graph/utils/node_utils.h"
#include "graph/utils/attr_utils.h"
#include "graph/debug/ge_attr_define.h"
#include "op_log.h"
#include "graph_optimizer/graph_fusion/fusion_pass_manager/fusion_pass_registry.h"
#include "pattern_fusion_util.h"

using namespace ge;
namespace fe {

static const std::string PATTERN_TRANSPOSE = "FusedNodeTranspose";
static const std::string PATTERN_RESHAPE = "FusedNodeReshape";

vector<FusionPattern*> TransposeReshapeFusionPass::DefinePatterns() {
  vector < FusionPattern * > patterns;

  FusionPattern* pattern = new(std::nothrow) FusionPattern("TransposeReshapeFusionPass");
  FUSION_PASS_CHECK(pattern == nullptr, OP_LOGE(FUSED_OP_TYPE.c_str(), "new a pattern object failed."),
           return patterns);

  pattern->AddOpDesc(PATTERN_TRANSPOSE, {"TransposeD"})
          .AddOpDesc(PATTERN_RESHAPE, {"Reshape"})
          .SetInputs(PATTERN_RESHAPE,
                     {PATTERN_TRANSPOSE})
          .SetOutput(PATTERN_RESHAPE);

  patterns.push_back(pattern);

  return patterns;
}

Status TransposeReshapeFusionPass::Fusion(ge::ComputeGraph &graph,
                                          Mapping &mapping,
                                          vector<ge::NodePtr> &fusionNodes)
{
  ge::NodePtr transNode = GetNodeFromMapping(PATTERN_TRANSPOSE, mapping);
  FUSION_PASS_CHECK(transNode == nullptr, OP_LOGE(FUSED_OP_TYPE.c_str(), "TransposeNode is null, fusion failed."), return PARAM_INVALID);
  ge::OpDescPtr transDesc = transNode->GetOpDesc();
  FUSION_PASS_CHECK(transDesc == nullptr, OP_LOGE(FUSED_OP_TYPE.c_str(), "TransposeNode's OpDesc is null, fusion failed."), return PARAM_INVALID);
  //TransposeD should only have one output anchor
  if (transNode->GetAllOutAnchors().size() != 1) {
    OP_LOGI(FUSED_OP_TYPE.c_str(), "The TransposeD node should only have 1 output anchor.");
    return NOT_CHANGED;
  }
  if (transNode->GetOutDataAnchor(0)->GetPeerInDataAnchors().size() != 1) {
    OP_LOGI(FUSED_OP_TYPE.c_str(), "The TransposeD node should only have 1 output anchor that link to other nodes.");
    return NOT_CHANGED;
  }
  if (transNode->GetOutDataAnchor(0)->GetPeerInDataAnchors().at(0)->GetOwnerNode()->GetType() != "Reshape") {
    OP_LOGI(FUSED_OP_TYPE.c_str(), "The TransposeD node should only have 1 output anchor that link to Reshape.");
    return NOT_CHANGED;
  }
  ge::NodePtr reshapeNode = GetNodeFromMapping(PATTERN_RESHAPE, mapping);
  FUSION_PASS_CHECK(reshapeNode == nullptr, OP_LOGE(FUSED_OP_TYPE.c_str(), "ReshapeNode is null, fusion failed."), return PARAM_INVALID);
  ge::OpDescPtr reshapeDesc = reshapeNode->GetOpDesc();
  FUSION_PASS_CHECK(reshapeDesc == nullptr, OP_LOGE(FUSED_OP_TYPE.c_str(), "ReshapeNode's OpDesc is null, fusion failed."), return PARAM_INVALID);
  if (reshapeNode->GetAllInDataAnchors().size() != 2) {
    OP_LOGI(FUSED_OP_TYPE.c_str(), "The Reshape node should only have 2 input anchor.");
    return NOT_CHANGED;
  }
  if (reshapeNode->GetAllOutDataAnchors().size() != 1) {
    OP_LOGI(FUSED_OP_TYPE.c_str(), "The Reshape node should only have 1 output anchor.");
    return NOT_CHANGED;
  }
  vector<int64_t> reshapeDimInfo = reshapeDesc->GetOutputDesc(0).GetOriginShape().GetDims();
  vector<int64_t> transposeDimInfo = transDesc->GetInputDesc(0).GetOriginShape().GetDims();
  if (reshapeDimInfo.size() == 1) {
    if (reshapeDimInfo[0] % 16 != 0) {
      OP_LOGI(FUSED_OP_TYPE.c_str(), "Node[%s]'s dimsize is [%zu], last one dimension should be divisible by 16, but actually is [%ld].",
              reshapeNode->GetName().c_str(), reshapeDimInfo.size(), reshapeDimInfo[0]);
      return NOT_CHANGED;
    }
  } else if (reshapeDimInfo.size() >= 2) {
    if (reshapeDimInfo[reshapeDimInfo.size() - 1] % 16 != 0 || reshapeDimInfo[reshapeDimInfo.size() - 2] % 16 != 0) {
      OP_LOGI(FUSED_OP_TYPE.c_str(), "Node[%s]'s dimsize is [%zu], last two dimension should be divisible by 16, but actually is [%ld] and [%ld].",
              reshapeNode->GetName().c_str(), reshapeDimInfo.size(),
              reshapeDimInfo[reshapeDimInfo.size() - 2], reshapeDimInfo[reshapeDimInfo.size() - 1]);
      return NOT_CHANGED;
    }
  } else {
    OP_LOGI(FUSED_OP_TYPE.c_str(), "Node[%s]'s dimsize is [%zu], cannot do fusion.", reshapeNode->GetName().c_str(), reshapeDimInfo.size());
    return NOT_CHANGED;
  }
  if (transposeDimInfo.size() == 1) {
    if (transposeDimInfo[0] % 16 != 0) {
      OP_LOGI(FUSED_OP_TYPE.c_str(), "Node[%s]'s dimsize is [%zu], last one dimension should be divisible by 16, but actually is [%ld].",
              transNode->GetName().c_str(), transposeDimInfo.size(), transposeDimInfo[0]);
      return NOT_CHANGED;
    }
  } else if (transposeDimInfo.size() >= 2) {
    if (transposeDimInfo[transposeDimInfo.size() - 1] % 16 != 0 || transposeDimInfo[transposeDimInfo.size() - 2] % 16 != 0) {
      OP_LOGI(FUSED_OP_TYPE.c_str(), "Node[%s]'s dimsize is [%zu], last two dimension should be divisible by 16, but actually is [%ld] and [%ld].",
              transNode->GetName().c_str(), transposeDimInfo.size(),
              transposeDimInfo[transposeDimInfo.size() - 2], transposeDimInfo[transposeDimInfo.size() - 1]);
      return NOT_CHANGED;
    }
  } else {
    OP_LOGI(FUSED_OP_TYPE.c_str(), "Node[%s]'s dimsize is [%zu], cannot do fusion.", transNode->GetName().c_str(), transposeDimInfo.size());
    return NOT_CHANGED;
  }
  vector<int32_t> permValue;
  ge::AttrUtils::GetListInt(transDesc, "perm", permValue);
  ge::AttrUtils::SetListInt(reshapeDesc, "perm", permValue);
  ge::AttrUtils::SetBool(reshapeDesc, "transpose_first", true);
  ge::GeTensorDesc reshapeOutputTensor = reshapeDesc->GetOutputDesc(0);
  ge::GeShape reshapeOutputShape = reshapeOutputTensor.GetShape();
  vector<int64_t> dimInfo = reshapeOutputShape.GetDims();
  reshapeDesc->SetName(reshapeDesc->GetName() + "/ConfusionTranspose");
  reshapeDesc->SetType("ConfusionTranspose");
  reshapeDesc->UpdateInputDesc(0, transDesc->GetInputDesc(0));
  // connect in control anchor
  if (transNode->GetInControlAnchor() != nullptr) {
    if (!transNode->GetInControlAnchor()->GetPeerOutControlAnchors().empty() && reshapeNode->GetInControlAnchor() != nullptr) {
      OP_LOGI(FUSED_OP_TYPE.c_str(), "The PeerOutControlAnchors of fused node[%s] input control anchor is empty.", transNode->GetName().c_str());
      for (OutControlAnchorPtr outCtrlAnchorPtr : transNode->GetInControlAnchor()->GetPeerOutControlAnchors()) {
        FUSION_PASS_CHECK(SUCCESS != ge::GraphUtils::AddEdge(outCtrlAnchorPtr, reshapeNode->GetInControlAnchor()),
                 OP_LOGE(FUSED_OP_TYPE.c_str(), "Fail to add input control edge for fusion node:%s.", reshapeNode->GetName().c_str()),
                 return FAILED);
      }
    }
    transNode->GetInControlAnchor()->UnlinkAll();
  }
  // connect out control anchor
  if (transNode->GetOutControlAnchor() != nullptr) {
    if (!transNode->GetOutControlAnchor()->GetPeerInControlAnchors().empty() && reshapeNode->GetOutControlAnchor() != nullptr) {
      OP_LOGI(FUSED_OP_TYPE.c_str(), "The PeerInControlAnchors of fused node[%s] output control anchor is empty.", transNode->GetName().c_str());
      for (InControlAnchorPtr inCtrlAnchorPtr : transNode->GetOutControlAnchor()->GetPeerInControlAnchors()) {
        FUSION_PASS_CHECK(SUCCESS != ge::GraphUtils::AddEdge(reshapeNode->GetOutControlAnchor(), inCtrlAnchorPtr),
                 OP_LOGE(FUSED_OP_TYPE.c_str(), "Fail to add output control edge for fusion node:%s.", reshapeNode->GetName().c_str()),
                 return FAILED);
      }
    }
    transNode->GetOutControlAnchor()->UnlinkAll();
  }
  if (transNode->GetAllInDataAnchors().size() != 1) {
    OP_LOGI(FUSED_OP_TYPE.c_str(), "Node:[%s] only should have one input, but actually have %d.", transNode->GetName().c_str(), transNode->GetAllInDataAnchors().size());
    return FAILED;
  }
  InDataAnchorPtr inAnchorPtr = reshapeNode->GetInDataAnchor(0);
  inAnchorPtr->UnlinkAll();
  FUSION_PASS_CHECK(SUCCESS != ge::GraphUtils::AddEdge(transNode->GetInDataAnchor(0)->GetPeerOutAnchor(),
                                              inAnchorPtr),
           OP_LOGE(FUSED_OP_TYPE.c_str(), "Add edge from fused node:%s to fusion node:%s failed.", transNode->GetName().c_str(), reshapeNode->GetName().c_str()),
           return FAILED);
  OP_LOGD(FUSED_OP_TYPE.c_str(), "Add edge from fused node:%s to fusion node:%s.", transNode->GetName().c_str(),  reshapeNode->GetName().c_str());
  for (auto inAnchorPtr : transNode->GetAllInDataAnchors()) {
    if (inAnchorPtr != nullptr) {
      inAnchorPtr->UnlinkAll();
    }
  }
  for (auto outAnchorPtr : transNode->GetAllOutDataAnchors()) {
    if (outAnchorPtr != nullptr) {
      outAnchorPtr->UnlinkAll();
    }
  }
  FUSION_PASS_CHECK(ge::GRAPH_SUCCESS != graph.RemoveNode(transNode), OP_LOGE(FUSED_OP_TYPE.c_str(), "Remove node:[%s] failed.", transDesc->GetName().c_str()), return FAILED);
  ge::NodePtr confusionTransposeD = nullptr;
  std::string fusionOpType = "ConfusionTransposeD";  //TileD为融合后的算子名称，需要与算子信息库中一致
  std::vector<PassAttrInfo> confusionTransposeAttrInfo;
  //SetListInt设置该属性为list还是单个值，对于不同数据类型的属性，提供 SetListInt、SetInt、SetListFloat、SetFloat 四种设置
  PassAttrInfo shape = {1, "shape", "SetListInt"}; //1和2分别表示需要转换的第一个和第二个const输入需要转换为属性（从0开始计数）
  confusionTransposeAttrInfo.push_back(shape);
  Status ret = PatternFusionUtil::ConstToAttrWithNode(graph, reshapeNode,
                   fusionOpType, confusionTransposeAttrInfo,
                   confusionTransposeD);
  if (ret != SUCCESS) {
    OP_LOGI(FUSED_OP_TYPE.c_str(), "Tile has input which is not a constant, graph not changed.");
    return NOT_CHANGED;
  }
  ge::AttrUtils::SetListInt(confusionTransposeD->GetOpDesc(), "shape", dimInfo);
  fusionNodes.push_back(confusionTransposeD);
  return SUCCESS;
}
REGISTER_PASS("TransposeReshapeFusionPass", BUILT_IN_GRAPH_PASS, TransposeReshapeFusionPass);  //第三个参数ConstToAttrTilePass与类名保持一致
}
