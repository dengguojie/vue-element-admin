/**
 * Copyright 2020 Huawei Technologies Co., Ltd
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
 * \file dw_group_fusion_pass.cc
 * \brief
 */
#include "dw_group_fusion_pass.h"
#include <vector>
#include <sstream>
#include "graph/utils/op_desc_utils.h"
#include "graph/utils/graph_utils.h"
#include "graph/utils/node_utils.h"
#include "graph/utils/attr_utils.h"
#include "graph/debug/ge_attr_define.h"
#include "op_log.h"
#include "pattern_fusion_util.h"
#include "graph_optimizer/graph_fusion/fusion_pass_manager/fusion_pass_registry.h"

using namespace ge;

namespace fe {
const string PATTERN_CONV2D_ID = "conv2d_group_id";
const string CONV2D_DW_TYPE = "Conv2DBackpropFilterD";
const string CONV2D_DX_TYPE = "Conv2DBackpropInputD";

vector<FusionPattern*> DwGroupFusionPass::DefinePatterns() {
  vector<FusionPattern*> patterns;
  FusionPattern* pattern = new (std::nothrow) FusionPattern("Conv2DGroupFusionPass");
  FUSION_PASS_CHECK(pattern == nullptr, OP_LOGE(FUSED_OP_TYPE.c_str(), "new a pattern object failed."),
                    return patterns);
  pattern->AddOpDesc(PATTERN_CONV2D_ID, {CONV2D_DW_TYPE, CONV2D_DX_TYPE}).SetOutput(PATTERN_CONV2D_ID);
  patterns.push_back(pattern);
  return patterns;
}

int64_t DwGroupFusionPass::GetGroups(ge::OpDescPtr &convDesc) {
  int64_t groups = 1;
  bool hasGroup = ge::AttrUtils::GetInt(convDesc, "groups", groups);
  return hasGroup ? groups : 1;
}

bool DwGroupFusionPass::GenerateSplitNode(ge::ComputeGraph &graph, ge::OpDescPtr &convDesc, int64_t &groups,
                                          ge::NodePtr &splitNode, ge::GeTensorDesc &splitOutDesc) {
  OpDescPtr sliceDesc;
  string convOpName = convDesc->GetName();
  GeTensorDesc inputDesc = convDesc->GetInputDesc(1);
  inputDesc.SetOriginDataType(DT_FLOAT16);
  inputDesc.SetDataType(DT_FLOAT16);
  GeShape inputShape = inputDesc.GetShape();
  size_t inChannelIdx = -1;
  FUSION_PASS_CHECK(SUCCESS != PatternFusionUtil::ParseChannelIdx(inputDesc, inChannelIdx),
            OP_LOGE(FUSED_OP_TYPE.c_str(),
                    "The original format of the conv node[name=%s, type=%s]'s input0 is %s, which is unsupportable.",
                    convDesc->GetName().c_str(), convDesc->GetType().c_str(),
                    ge::TypeUtils::FormatToSerialString(inputDesc.GetFormat()).c_str()),
            return FAILED);
  int newInputChn = inputShape.GetDim(inChannelIdx);
  GeShape splitOutShape = inputShape;
  splitOutShape.SetDim(inChannelIdx, newInputChn / groups);

  FUSION_PASS_MAKE_SHARED((sliceDesc = std::make_shared<ge::OpDesc>(convOpName+"_split", "SplitD")), return FAILED);
  AttrUtils::SetInt(sliceDesc, "split_dim", inChannelIdx);
  AttrUtils::SetInt(sliceDesc, "num_split", groups);
  splitOutDesc = inputDesc;
  splitOutDesc.Update(splitOutShape, inputDesc.GetOriginFormat(), DT_FLOAT16);
  splitOutDesc.SetOriginShape(splitOutShape);
  sliceDesc->AddInputDesc(inputDesc);
  for (int i = 0; i < groups; i++) {
    sliceDesc->AddOutputDesc(splitOutDesc);
  }
  splitNode = graph.AddNode(sliceDesc);
  return true;
}

bool DwGroupFusionPass::GenerateNewConvNodes(ge::ComputeGraph &graph, ge::OpDescPtr &convDesc,
                                             const ge::GeTensorDesc &splitOutDesc, vector<ge::NodePtr> &newConvNodes,
                                             ge::GeTensorDesc &newConvOutDesc) {
  int64_t groups = GetGroups(convDesc);

  GeTensorDesc inDesc = splitOutDesc;
  inDesc.SetOriginDataType(convDesc->GetInputDesc(0).GetOriginDataType());
  inDesc.SetDataType(convDesc->GetInputDesc(0).GetDataType());

  string convOpName = convDesc->GetName();
  GeTensorDesc outputDesc = convDesc->GetOutputDesc(0);
  GeShape newConvOutShape = outputDesc.GetShape();
  size_t outChannelIdx = -1;
  if (convDesc->GetType() == CONV2D_DX_TYPE) {
    FUSION_PASS_CHECK(SUCCESS != PatternFusionUtil::ParseChannelIdx(outputDesc, outChannelIdx),
        OP_LOGE(FUSED_OP_TYPE.c_str(),
                "The original format of the conv node[name=%s, type=%s]'s input0 is %s, which is unsupportable.",
                convDesc->GetName().c_str(), convDesc->GetType().c_str(),
                ge::TypeUtils::FormatToSerialString(outputDesc.GetFormat()).c_str()),
        return FAILED);
  } else {
    FUSION_PASS_CHECK(SUCCESS != PatternFusionUtil::ParseNChannelIdx(outputDesc, outChannelIdx),
        OP_LOGE(FUSED_OP_TYPE.c_str(),
                "The original format of the conv node[name=%s, type=%s]'s input0 is %s, which is unsupportable.",
                convDesc->GetName().c_str(), convDesc->GetType().c_str(),
                ge::TypeUtils::FormatToSerialString(outputDesc.GetFormat()).c_str()),
        return FAILED);
  }
  int newConvOutChn = newConvOutShape.GetDim(outChannelIdx) / groups;
  newConvOutShape.SetDim(outChannelIdx, newConvOutChn);
  newConvOutDesc = outputDesc;
  newConvOutDesc.Update(newConvOutShape, outputDesc.GetOriginFormat(), convDesc->GetOutputDesc(0).GetDataType());
  newConvOutDesc.SetOriginShape(newConvOutShape);
  newConvOutDesc.SetOriginDataType(convDesc->GetOutputDesc(0).GetOriginDataType());
  for (int64_t i = 0; i < groups; i++) {
    ostringstream newConvName;
    newConvName << convOpName << "_" << i;
    OpDescPtr newConvDesc = AttrUtils::CopyOpDesc(convDesc);
    FUSION_PASS_CHECK(newConvDesc == nullptr,
                      OP_LOGE(FUSED_OP_TYPE.c_str(), "Node:%s's OpDesc is null, fusion failed.",
                              convDesc->GetName().c_str()),
                      return PARAM_INVALID);
    newConvDesc->SetName(newConvName.str());
    (void)newConvDesc->UpdateInputDesc(0, inDesc);
    for (unsigned int j = 1; j < convDesc->GetAllInputsDesc().size(); j++) {
      newConvDesc->UpdateInputDesc(j, inDesc);
    }
    FUSION_PASS_CHECK(ge::GRAPH_SUCCESS != newConvDesc->UpdateOutputDesc(0, newConvOutDesc),
             OP_LOGE("Update node:%s's 1st outputfailed.", convDesc->GetName().c_str()),
             return false);
    AttrUtils::SetInt(newConvDesc, "groups", 1);
    std::string attr_name;
    if (newConvDesc->GetType() == CONV2D_DW_TYPE) {
      attr_name = "filter_size";
    } else {
      attr_name = "input_size";
    }
    AttrUtils::SetListInt(newConvDesc, attr_name, newConvOutShape.GetDims());
    NodePtr newConvNode = graph.AddNode(newConvDesc);
    newConvNodes.push_back(newConvNode);
  }
  return true;
}

bool DwGroupFusionPass::GenerateConcatNode(ge::ComputeGraph &graph, ge::OpDescPtr &convDesc, const int64_t &groups,
                                           ge::GeTensorDesc &newConvOutDesc, ge::NodePtr &concatNode) {
  string convOpName = convDesc->GetName();
  OpDescPtr concatDesc;
  newConvOutDesc.SetOriginDataType(DT_FLOAT16);
  newConvOutDesc.SetDataType(DT_FLOAT16);
  FUSION_PASS_MAKE_SHARED((concatDesc = std::make_shared<ge::OpDesc>(convOpName+"_concat", "ConcatD")), return false);
  for (int i = 0; i < groups; i++) {
    concatDesc->AddInputDesc(newConvOutDesc);
  }
  concatDesc->AddOutputDesc(convDesc->GetOutputDesc(0));
  GeTensorDesc inputDesc = convDesc->GetOutputDesc(0);
  size_t outChannelIdx = -1;
  if (convDesc->GetType() == CONV2D_DX_TYPE) {
    FUSION_PASS_CHECK(SUCCESS != PatternFusionUtil::ParseChannelIdx(outputDesc, outChannelIdx),
        OP_LOGE(FUSED_OP_TYPE.c_str(),
                "The original format of the conv node[name=%s, type=%s]'s input0 is %s, which is unsupportable.",
                convDesc->GetName().c_str(), convDesc->GetType().c_str(),
                ge::TypeUtils::FormatToSerialString(outputDesc.GetFormat()).c_str()),
        return FAILED);
  } else {
    FUSION_PASS_CHECK(SUCCESS != PatternFusionUtil::ParseNChannelIdx(outputDesc, outChannelIdx),
        OP_LOGE(FUSED_OP_TYPE.c_str(),
                "The original format of the conv node[name=%s, type=%s]'s input0 is %s, which is unsupportable.",
                convDesc->GetName().c_str(), convDesc->GetType().c_str(),
                ge::TypeUtils::FormatToSerialString(outputDesc.GetFormat()).c_str()),
        return FAILED);
  }
  AttrUtils::SetInt(concatDesc, "concat_dim", outChannelIdx); // c axis concat
  concatNode = graph.AddNode(concatDesc);
  return true;
}

bool DwGroupFusionPass::Relink(ge::NodePtr &convNode, ge::NodePtr &splitNode, vector<ge::NodePtr> &newConvNodes,
                               ge::NodePtr &concatNode) {
  auto op_desc = convNode->GetOpDesc();
  int64_t groups = GetGroups(op_desc);

  Node::Vistor<NodePtr> inNodes = convNode->GetInDataNodes();
  FUSION_PASS_CHECK(inNodes.size() < 2,
           OP_LOGE(FUSED_OP_TYPE.c_str(), "conv input nodes num(%d) < 2", inNodes.size()), return false);

  Node::Vistor<NodePtr> outNodes = convNode->GetOutDataNodes();
  auto peer_out_anchor = convNode->GetInDataAnchor(0)->GetPeerOutAnchor();
  if (peer_out_anchor == nullptr) {
    OP_LOGE(FUSED_OP_TYPE.c_str(), "conv node[%s]'s peer out anchor is empty.", convNode->GetName().c_str());
    return false;
  }
  vector<int> outAnchorIndexes;
  for (size_t i = 0; i < outNodes.size(); i++) {
    outAnchorIndexes.push_back(convNode->GetOutDataAnchor(0)->GetPeerInDataAnchors().at(i)->GetIdx());
  }
  for (auto outAnchor : convNode->GetAllOutDataAnchors()) {
    if (outAnchor != nullptr) {
      outAnchor->UnlinkAll();
    }
  }

  graphStatus status = GraphUtils::AddEdge(peer_out_anchor, splitNode->GetInDataAnchor(0));
  FUSION_PASS_CHECK(status != GRAPH_SUCCESS,
          OP_LOGE(FUSED_OP_TYPE.c_str(), "add data to slice edge fail"), return false);
  for (int i = 0; i < groups; i++) {
    status = GraphUtils::AddEdge(splitNode->GetOutDataAnchor(i), newConvNodes[i]->GetInDataAnchor(1));
    FUSION_PASS_CHECK(status != GRAPH_SUCCESS,
            OP_LOGE(FUSED_OP_TYPE.c_str(), "add slice to conv edge fail"), return false);
    status = GraphUtils::AddEdge(newConvNodes[i]->GetOutDataAnchor(0), concatNode->GetInDataAnchor(i));
    FUSION_PASS_CHECK(status != GRAPH_SUCCESS,
            OP_LOGE(FUSED_OP_TYPE.c_str(), "add conv to concat edge fail"), return false);
  }

  for (size_t i = 0; i < outNodes.size(); i++) {
    status = GraphUtils::AddEdge(concatNode->GetOutAnchor(0),
                                 outNodes.at(i)->GetInDataAnchor(outAnchorIndexes[i]));
    FUSION_PASS_CHECK(status != GRAPH_SUCCESS,
            OP_LOGE(FUSED_OP_TYPE.c_str(), "add concat to output edge fail"), return false);
  }
  return true;
}

Status DwGroupFusionPass::ProcessGroupConv(ge::ComputeGraph &graph, ge::NodePtr &convNode) {
  NodePtr splitNode = nullptr;
  GeTensorDesc splitOutDesc;
  OpDescPtr convDesc = convNode->GetOpDesc();
  int64_t groups = GetGroups(convDesc);
  FUSION_PASS_CHECK(!GenerateSplitNode(graph, convDesc, groups, splitNode, splitOutDesc),
           OP_LOGE(FUSED_OP_TYPE.c_str(), "generate split node fail"), return FAILED);

  vector<NodePtr> newConvNodes;
  GeTensorDesc newConvOutDesc;
  FUSION_PASS_CHECK(!GenerateNewConvNodes(graph, convDesc, splitOutDesc, newConvNodes, newConvOutDesc),
           OP_LOGE(FUSED_OP_TYPE.c_str(), "generate new conv nodes fail"), return FAILED);

  NodePtr concatNode;
  FUSION_PASS_CHECK(!GenerateConcatNode(graph, convDesc, groups, newConvOutDesc, concatNode),
           OP_LOGE(FUSED_OP_TYPE.c_str(), "generate concat fail"), return FAILED);

  FUSION_PASS_CHECK(!Relink(convNode, splitNode, newConvNodes, concatNode),
           OP_LOGE(FUSED_OP_TYPE.c_str(), "relink fail"), return FAILED);

  OP_LOGD(FUSED_OP_TYPE.c_str(), "conv[%s]'s weight size is [%zu]",
          convNode->GetName().c_str(), OpDescUtils::GetWeights(convNode).size());
  // find sliceD of conv's 2nd or 3rd, 4th input
  size_t inChannelIdx = -1;
  for (unsigned int i = 0; i < convNode->GetInDataNodes().size() - 1; i++) {
    GeTensorDesc inputDesc = convDesc->GetInputDesc(i);
    if (convDesc->GetType() == CONV2D_DX_TYPE) {
      FUSION_PASS_CHECK(SUCCESS != PatternFusionUtil::ParseNChannelIdx(inputDesc, inChannelIdx),
          OP_LOGE(FUSED_OP_TYPE.c_str(),
                  "The original format of the conv node[name=%s, type=%s]'s input0 is %s, which is unsupportable.",
                  convDesc->GetName().c_str(), convDesc->GetType().c_str(),
                  ge::TypeUtils::FormatToSerialString(inputDesc.GetFormat()).c_str()),
          return FAILED);
    } else {
      FUSION_PASS_CHECK(SUCCESS != PatternFusionUtil::ParseChannelIdx(inputDesc, inChannelIdx),
          OP_LOGE(FUSED_OP_TYPE.c_str(),
                 "The original format of the conv node[name=%s, type=%s]'s input0 is %s, which is unsupportable.",
                 convDesc->GetName().c_str(), convDesc->GetType().c_str(),
                 ge::TypeUtils::FormatToSerialString(inputDesc.GetFormat()).c_str()),
          return FAILED);
    }
    if (inputDesc.GetShape().GetDimNum() == 1) {
      inChannelIdx = 0;
    }
    FUSION_PASS_CHECK(PatternFusionUtil::InsertSliceDNodes(graph, convNode, i,
                                                  newConvNodes, groups, inChannelIdx) != SUCCESS,
             OP_LOGE(FUSED_OP_TYPE.c_str(), "Insert SliceD node between node[%s] and weight node falied.",
                     convNode->GetName().c_str()),
             return FAILED);
  }

  for (auto inAnchor : convNode->GetAllInDataAnchors()) {
    if (inAnchor != nullptr) {
      inAnchor->UnlinkAll();
    }
  }
  FUSION_PASS_CHECK(SUCCESS != graph.RemoveNode(convNode),
           OP_LOGE(FUSED_OP_TYPE.c_str(), "RemoveNode %s failed.", convNode->GetName().c_str()),
           return FAILED);
  return SUCCESS;
}

Status DwGroupFusionPass::Fusion(ge::ComputeGraph& graph, Mapping& mapping, vector<ge::NodePtr>& newNodes) {
  OP_LOGI(FUSED_OP_TYPE.c_str(), "Enter Conv2DGroupPass::Fusion.");
  NodePtr convNode = GetNodeFromMapping(PATTERN_CONV2D_ID, mapping);
  OpDescPtr convDesc = convNode->GetOpDesc();

  // 1.if the deconv node doesn't have the attribute groups or the value is 1, just return not changed.
  int64_t groups = 1;
  bool hasGroup = ge::AttrUtils::GetInt(convDesc, "groups", groups);
  if (!hasGroup || groups == 1) {
    OP_LOGD(FUSED_OP_TYPE.c_str(),
            "The conv node[name=%s, type=%s] doesn't have the attribute groups, or the value is 1.",
            convDesc->GetName().c_str(), convDesc->GetType().c_str());
    return NOT_CHANGED;
  }

  GeTensorDesc inputDesc = convDesc->GetInputDesc(1);
  size_t inChannelIdx = -1;
  FUSION_PASS_CHECK(
      SUCCESS != PatternFusionUtil::ParseChannelIdx(inputDesc, inChannelIdx),
      OP_LOGW(FUSED_OP_TYPE.c_str(),
              "The original format of the conv node[name=%s, type=%s]'s input0 is %s, which is unsupportable.",
              convDesc->GetName().c_str(), convDesc->GetType().c_str(),
              ge::TypeUtils::FormatToSerialString(inputDesc.GetFormat()).c_str()),
      return NOT_CHANGED);
  int64_t inChn = inputDesc.GetOriginShape().GetDim(inChannelIdx);

  GeTensorDesc outputDesc = convDesc->GetOutputDesc(0);
  size_t outChannelIdx = -1;
  if (convDesc->GetType() == CONV2D_DX_TYPE) {
    FUSION_PASS_CHECK(SUCCESS != PatternFusionUtil::ParseChannelIdx(outputDesc, outChannelIdx),
          OP_LOGW(FUSED_OP_TYPE.c_str(),
                  "The original format of the conv node[name=%s, type=%s]'s input0 is %s, which is unsupportable.",
                  convDesc->GetName().c_str(), convDesc->GetType().c_str(),
                  ge::TypeUtils::FormatToSerialString(outputDesc.GetFormat()).c_str()),
          return NOT_CHANGED);
  } else {
    FUSION_PASS_CHECK(SUCCESS != PatternFusionUtil::ParseNChannelIdx(outputDesc, outChannelIdx),
          OP_LOGW(FUSED_OP_TYPE.c_str(),
                  "The original format of the conv node[name=%s, type=%s]'s input0 is %s, which is unsupportable.",
                  convDesc->GetName().c_str(), convDesc->GetType().c_str(),
                  ge::TypeUtils::FormatToSerialString(outputDesc.GetFormat()).c_str()),
          return NOT_CHANGED);
  }
  int64_t outChn = outputDesc.GetOriginShape().GetDim(outChannelIdx);
  if (PatternFusionUtil::IsUnknownShape(outChn) || PatternFusionUtil::IsUnknownShape(inChn)) {
    OP_LOGW(FUSED_OP_TYPE.c_str(), "Conv2DGroupFusionPass cannot be applied for unknown shape.");
    return NOT_CHANGED;
  }
  if (inChn % groups == 0 && outChn % groups == 0) {
    return ProcessGroupConv(graph, convNode);
  } else {
    OP_LOGW(FUSED_OP_TYPE.c_str(),
            "The number of input channel(%lld) or output channel(%lld) of "
            "the conv node[name=%s, type=%s] is not divisible by groups(%lld)",
            inChn, outChn, convDesc->GetName().c_str(), convDesc->GetType().c_str(), groups);
    return NOT_CHANGED;
  }
}
REGISTER_PASS("ZDwGroupFusionPass", BUILT_IN_GRAPH_PASS, DwGroupFusionPass);
}  // namespace fe
