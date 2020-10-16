/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2019-2019. All rights reserved.
 *
 * @brief padd conv2d fusion pass
 *
 */

#include <memory>
#include <string>
#include "padd_conv2d_fusion_pass.h"
#include "graph/debug/ge_attr_define.h"
#include "graph_optimizer/graph_fusion/fusion_pass_manager/fusion_pass_registry.h"
#include "op_log.h"
#include "pattern_fusion_util.h"
#include "graph/utils/graph_utils.h"


namespace fe {

static const string PATTERN_INPUTS1 = "input1";
static const string PATTERN_PADD = "padd";
static const string PATTERN_CONV2D = "conv2d";
static const string PADD = "PadD";
static const string PADDINGS = "paddings";
static const string PADS = "pads";
static const string PADDING = "padding";
static const string CONV2D = "Conv2D";
static const string INPUT_SIZE = "input_size";
static const string CONV2DBACKPROPFILTERD = "Conv2DBackpropFilterD";
static const string FUSEBATCHNORMGRADD = "BNTrainingReduceGrad";
static const string CONV2DBACKPROPINPUTD = "Conv2DBackpropInputD";
static const string SLICE = "SliceD";
static const int DIM_NUM4 = 4;
static const int DIRECTION_COUNT = 2;
vector<FusionPattern *> PaddConv2dFusionPass::DefinePatterns() {
  OP_LOGI(FUSED_OP_TYPE.c_str(), "Define PaddConv2dFusionPass pattern begin");
  vector<FusionPattern *> patterns;
  FusionPattern *pattern =
      new (std::nothrow) FusionPattern("PaddConv2dFusionPass");

  FUSION_PASS_CHECK(pattern == nullptr, OP_LOGE(FUSED_OP_TYPE.c_str(), "New a pattern object failed."),
           return patterns);

  pattern->AddOpDesc(PATTERN_PADD, {PADD})
      .AddOpDesc(PATTERN_CONV2D, {CONV2D})
      .AddOpDesc(PATTERN_INPUTS1)
      .SetInputs(PATTERN_CONV2D, {PATTERN_PADD, PATTERN_INPUTS1})
      .SetOutput(PATTERN_CONV2D);
  patterns.push_back(pattern);
  OP_LOGI(FUSED_OP_TYPE.c_str(), "Define PaddConv2dFusionPass pattern end");
  return patterns;
}

Status PaddConv2dFusionPass::Fusion(ge::ComputeGraph &graph,
                                    Mapping &mapping,
                                    vector<ge::NodePtr> &fusionNodes)
{
  OP_LOGI(FUSED_OP_TYPE.c_str(), "Define PaddConv2dFusionPass fusion begin");
  ge::NodePtr paddNode = GetNodeFromMapping(PATTERN_PADD, mapping);
  FUSION_PASS_CHECK(paddNode == nullptr, OP_LOGE(FUSED_OP_TYPE.c_str(), "padD Node is null, fusion failed."),
           return PARAM_INVALID);

  ge::NodePtr conv2dNode = GetNodeFromMapping(PATTERN_CONV2D, mapping);
  FUSION_PASS_CHECK(conv2dNode == nullptr,
           OP_LOGE(FUSED_OP_TYPE.c_str(), "Conv2D Node is null, fusion failed."),
           return PARAM_INVALID);

  int64_t convCount = 0;
  int64_t dwCount = 0;
  for (auto peerInDataAnchor : paddNode->GetOutDataAnchor(0)->GetPeerInDataAnchors()) {
    ge::NodePtr nextNode = peerInDataAnchor->GetOwnerNode();
    if (nextNode->GetType() == CONV2D) {
      convCount++;
    }
    if (nextNode->GetType() == CONV2DBACKPROPFILTERD) {
      dwCount++;
    }
  }
  FUSION_PASS_CHECK(convCount > 1,
           OP_LOGI(FUSED_OP_TYPE.c_str(), "Padnode have multiple conv2d outputs, can not fusion."),
           return NOT_CHANGED);
  FUSION_PASS_CHECK(dwCount > 1,
           OP_LOGI(FUSED_OP_TYPE.c_str(), "Padnode have multiple dw outputs, can not fusion."),
           return NOT_CHANGED);
  vector<vector<int64_t>> paddings;
  FUSION_PASS_CHECK(
      !ge::AttrUtils::GetListListInt(paddNode->GetOpDesc(), PADDINGS, paddings),
      OP_LOGI(FUSED_OP_TYPE.c_str(), "Get paddings attr failed."), return NOT_CHANGED);

  if (paddings.size() < DIM_NUM4 || paddings[0].size() < DIRECTION_COUNT ||
      paddings[1].size() < DIRECTION_COUNT ||
      paddings[2].size() < DIRECTION_COUNT ||
      paddings[3].size() < DIRECTION_COUNT) {
    OP_LOGI(FUSED_OP_TYPE.c_str(), "The number of paddings not valid, can not fusion.");
    return NOT_CHANGED;
  }

  int64_t paddingsT;
  int64_t paddingsB;
  int64_t paddingsL;
  int64_t paddingsR;
  if (paddNode->GetOpDesc()->GetInputDesc(0).GetFormat() == ge::FORMAT_NCHW) {
    paddingsT = paddings[2][0];
    paddingsB = paddings[2][1];
    paddingsL = paddings[3][0];
    paddingsR = paddings[3][1];
  } else if (paddNode->GetOpDesc()->GetInputDesc(0).GetFormat() ==
             ge::FORMAT_NHWC) {
    paddingsT = paddings[1][0];
    paddingsB = paddings[1][1];
    paddingsL = paddings[2][0];
    paddingsR = paddings[2][1];
  } else {
    OP_LOGI(FUSED_OP_TYPE.c_str(), "Padd intput Format is not NCHW or NHWC, can not fusion.");
    return NOT_CHANGED;
  }

  if (paddingsT < 0 || paddingsT > 255 || paddingsB < 0 || paddingsB > 255 ||
      paddingsL < 0 || paddingsL > 255 || paddingsR < 0 || paddingsR > 255) {
    OP_LOGI(FUSED_OP_TYPE.c_str(), "Paddings value not in [0,255], can not fusion.");
    return NOT_CHANGED;
  }
  ge::NodePtr kernelNode =
      conv2dNode->GetInDataAnchor(0)->GetPeerOutAnchor()->GetOwnerNode();
  if (kernelNode->GetOpDesc()->GetOutputDesc(0).GetFormat() ==
          ge::FORMAT_NCHW &&
      (kernelNode->GetOpDesc()->GetOutputDesc(0).GetShape().GetDim(2) <=
           paddingsT ||
       kernelNode->GetOpDesc()->GetOutputDesc(0).GetShape().GetDim(2) <=
           paddingsB)) {
    OP_LOGI(FUSED_OP_TYPE.c_str(), "Filter_H more than pad_H, can not fusion.");
    return NOT_CHANGED;
  }

  if (kernelNode->GetOpDesc()->GetOutputDesc(0).GetFormat() ==
          ge::FORMAT_HWCN &&
      (kernelNode->GetOpDesc()->GetOutputDesc(0).GetShape().GetDim(0) <=
           paddingsT ||
       kernelNode->GetOpDesc()->GetOutputDesc(0).GetShape().GetDim(0) <=
           paddingsB)) {
    OP_LOGI(FUSED_OP_TYPE.c_str(), "Filter_H more than pad_H, can not fusion.");
    return NOT_CHANGED;
  }

  vector<int64_t> pads;
  pads.push_back(paddingsT);
  pads.push_back(paddingsB);
  pads.push_back(paddingsL);
  pads.push_back(paddingsR);
  if (!paddNode->GetOutControlAnchor()->GetPeerInControlAnchors().empty()) {
    OP_LOGI(FUSED_OP_TYPE.c_str(), "PaddNode has control edge, can not fusion.");
    return NOT_CHANGED;
  }

  //Get conv2DBackpropFilterDNode and check the graph
  ge::NodePtr conv2DBackpropFilterDNode = nullptr;
  for (auto inDataAnchor :
      paddNode->GetOutDataAnchor(0)->GetPeerInDataAnchors()) {
    if (inDataAnchor->GetOwnerNode()->GetOpDesc()->GetType()
        == CONV2DBACKPROPFILTERD) {
      conv2DBackpropFilterDNode = inDataAnchor->GetOwnerNode();
    }
    FUSION_PASS_CHECK(inDataAnchor->GetOwnerNode()->GetOpDesc()->GetType() != CONV2D
                 && inDataAnchor->GetOwnerNode()->GetOpDesc()->GetType()
                     != CONV2DBACKPROPFILTERD,
             OP_LOGI(FUSED_OP_TYPE.c_str(), "Output node is not Conv2D or Conv2DBackpropFilterD, can not fusion."),
             return NOT_CHANGED);
  }
  //Get BatchnormGradNode and check the graph
  if (conv2DBackpropFilterDNode != nullptr) {
    ge::NodePtr BatchNormGradNode = nullptr;
    if (conv2DBackpropFilterDNode->GetInDataAnchor(1)->GetPeerOutAnchor()
        ->GetOwnerNode()->GetOpDesc()->GetType() == FUSEBATCHNORMGRADD) {
        BatchNormGradNode = conv2DBackpropFilterDNode
                                      ->GetInDataAnchor(1)
                                      ->GetPeerOutAnchor()
                                      ->GetOwnerNode();
    }
    if (BatchNormGradNode != nullptr) {
      ge::NodePtr conv2DbackpropinputNode = nullptr;
      for (auto inDataAnchor :
           BatchNormGradNode->GetOutDataAnchor(0)->GetPeerInDataAnchors()) {
        if (inDataAnchor->GetOwnerNode()->GetOpDesc()->GetType()
          == CONV2DBACKPROPINPUTD) {
          conv2DbackpropinputNode = inDataAnchor->GetOwnerNode();
        }
      }
      if (conv2DbackpropinputNode != nullptr) {
        //Get sliceNode and check the graph
        ge::NodePtr sliceNode = nullptr;
        int flag_slice = 0;
        for (auto inDataAnchor :
             conv2DbackpropinputNode->GetOutDataAnchor(0)->GetPeerInDataAnchors()) {
          if (inDataAnchor->GetOwnerNode()->GetType() == SLICE) {
            sliceNode = inDataAnchor->GetOwnerNode();
          }
          if (inDataAnchor->GetOwnerNode()->GetOpDesc()->GetType() != SLICE) {
            flag_slice = 1;
          }
        }
        if (sliceNode != nullptr && flag_slice == 0) {
          FUSION_PASS_CHECK(conv2DbackpropinputNode->GetOpDesc()->UpdateOutputDesc(
            0, sliceNode->GetOpDesc()->GetOutputDesc(0)) != SUCCESS,
            OP_LOGE(FUSED_OP_TYPE.c_str(), "Update output failed."), return FAILED);
          // change out edge of conv2dbackpropinput to slice
            FUSION_PASS_CHECK(ge::GraphUtils::RemoveEdge(
                sliceNode->GetInDataAnchor(0)->GetPeerOutAnchor(),
                sliceNode->GetInDataAnchor(0)) != SUCCESS,
                OP_LOGE(FUSED_OP_TYPE.c_str(), "Remove slice input0 edge error"), return FAILED);
            FUSION_PASS_CHECK(!ge::AttrUtils::SetListInt(conv2DbackpropinputNode->GetOpDesc(),
                PADS, pads), OP_LOGE(FUSED_OP_TYPE.c_str(), "Set paddings to %s failed.",
                conv2DbackpropinputNode->GetName().c_str()), return FAILED);
            FUSION_PASS_CHECK(!ge::AttrUtils::SetStr(conv2DbackpropinputNode->GetOpDesc(),
                PADDING, "SAME"),
            OP_LOGE(FUSED_OP_TYPE.c_str(), "Set padding attr failed."), return FAILED);
            vector<int64_t> input_size = sliceNode->GetOpDesc()
                                         ->GetOutputDesc(0).GetShape().GetDims();
            FUSION_PASS_CHECK(!ge::AttrUtils::SetListInt(conv2DbackpropinputNode->GetOpDesc(),
                                                INPUT_SIZE, input_size),
                     OP_LOGE(FUSED_OP_TYPE.c_str(), "Set input_size to %s failed.",
                             conv2DbackpropinputNode->GetName().c_str()),
                     return FAILED);

            // remove sliceNode output
            for (auto outDataAnchor :
                 sliceNode->GetOutDataAnchor(0)->GetPeerInDataAnchors()) {
              FUSION_PASS_CHECK(ge::GraphUtils::RemoveEdge(sliceNode->GetOutDataAnchor(0),
                  outDataAnchor) != SUCCESS,
                  OP_LOGE(FUSED_OP_TYPE.c_str(), "Remove out data edge failed."), return FAILED);
              FUSION_PASS_CHECK(ge::GraphUtils::AddEdge(conv2DbackpropinputNode->
                       GetOutDataAnchor(0), outDataAnchor) != SUCCESS,
                  OP_LOGE(FUSED_OP_TYPE.c_str(), "Add edge between node %s. and node %s failed.",
                      conv2DbackpropinputNode->GetName().c_str(),
                      outDataAnchor->GetOwnerNode()->GetName().c_str()),
                   return FAILED);
            }
            FUSION_PASS_CHECK(graph.RemoveNode(sliceNode) != SUCCESS,
            OP_LOGE(FUSED_OP_TYPE.c_str(), "Remove slice node failed."), return FAILED);
        }
      }
    }
  }
  vector<ge::NodePtr> nodeVector;
  nodeVector.push_back(conv2dNode);
  if (conv2DBackpropFilterDNode != nullptr) {
    nodeVector.push_back(conv2DBackpropFilterDNode);
  }
  for (ge::NodePtr nodePtr : nodeVector) {
    string nodeName = nodePtr->GetOpDesc()->GetType();
    // update input desc
    FUSION_PASS_CHECK(nodePtr->GetOpDesc()->UpdateInputDesc(
        0, paddNode->GetOpDesc()->GetInputDesc(0)) != SUCCESS,
             OP_LOGE(FUSED_OP_TYPE.c_str(), "Update %s input failed.", nodeName.c_str()), return FAILED);
    // change input edge of padd to conv2d/conv2DBackpropFilterD
    FUSION_PASS_CHECK(ge::GraphUtils::RemoveEdge(
        nodePtr->GetInDataAnchor(0)->GetPeerOutAnchor(),
        nodePtr->GetInDataAnchor(0)) != SUCCESS,
             OP_LOGE(FUSED_OP_TYPE.c_str(), "Remove %s input0 edge error", nodeName.c_str()), return FAILED);
    FUSION_PASS_CHECK(
        ge::GraphUtils::AddEdge(paddNode->GetInDataAnchor(0)->GetPeerOutAnchor(),
                                nodePtr->GetInDataAnchor(0)) != SUCCESS,
        OP_LOGE(FUSED_OP_TYPE.c_str(), "Add edge between node %s. and node %s failed.",
                paddNode->GetInDataAnchor(0)
                    ->GetPeerOutAnchor()
                    ->GetOwnerNode()
                    ->GetName()
                    .c_str(),
                nodePtr->GetName().c_str()),
        return FAILED);
    FUSION_PASS_CHECK(!ge::AttrUtils::SetListInt(nodePtr->GetOpDesc(), PADS, pads),
             OP_LOGE(FUSED_OP_TYPE.c_str(), "Set paddings to %s failed.", nodeName.c_str()), return FAILED);
    FUSION_PASS_CHECK(!ge::AttrUtils::SetStr(nodePtr->GetOpDesc(), PADDING, "SAME"),
             OP_LOGE(FUSED_OP_TYPE.c_str(), "Set padding attr failed."), return FAILED);
  }
  // remove paddNode output
  for (auto inDataAnchor :
       paddNode->GetOutDataAnchor(0)->GetPeerInDataAnchors()) {
    FUSION_PASS_CHECK(ge::GraphUtils::RemoveEdge(paddNode->GetOutDataAnchor(0),
                                        inDataAnchor) != SUCCESS,
             OP_LOGE(FUSED_OP_TYPE.c_str(), "Remove out data edge failed."), return FAILED);
  }
  if (paddNode->GetOutControlAnchor()) {
    for (auto inControlAnchor :
         paddNode->GetOutControlAnchor()->GetPeerInControlAnchors()) {
      FUSION_PASS_CHECK(ge::GraphUtils::RemoveEdge(paddNode->GetOutControlAnchor(),
                                          inControlAnchor) != SUCCESS,
               OP_LOGE(FUSED_OP_TYPE.c_str(), "Remove out control edge failed."), return FAILED);
    }
  }
  FUSION_PASS_CHECK(graph.RemoveNode(paddNode) != SUCCESS,
           OP_LOGE(FUSED_OP_TYPE.c_str(), "Remove PadD node failed."), return FAILED);
  fusionNodes.push_back(conv2dNode);

  OP_LOGI(FUSED_OP_TYPE.c_str(), "Define PaddConv2dFusionPass fusion end");
  return SUCCESS;
}
REGISTER_PASS("PaddConv2dFusionPass", BUILT_IN_GRAPH_PASS,
              PaddConv2dFusionPass);
}  // namespace fe
