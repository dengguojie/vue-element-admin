/**
 * Copyright 2019 Huawei Technologies Co., Ltd
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
 * \file conv_fusion_pass_base.cpp
 * \brief fuse conv batchnorm, conv scale, batchnorm conv
 */
#include "conv_fusion_pass_base.h"
#include <climits>
#include <iostream>
#include <map>
#include <memory>
#include <set>
#include <vector>
#include <string>
#include "graph/debug/ge_attr_define.h"
#include "graph/utils/graph_utils.h"
#include "graph/utils/node_utils.h"
#include "graph/utils/op_desc_utils.h"
#include "graph/utils/tensor_utils.h"
#include "graph_optimizer/graph_fusion/fusion_pass_manager/fusion_pass_registry.h"
#include "pattern_fusion_util.h"
#include "common/util/error_manager/error_manager.h"
#include "../../../op_proto/util/error_util.h"

namespace fe {
Status ConvFusionPassBase::DoFusion(ge::ComputeGraph& graph, ge::NodePtr convNode, ge::NodePtr destNode,
                                    vector<ge::NodePtr>& fusionNodes) {
  string convNodeName = convNode->GetName();
  // isolate node, to be deleted
  Status removeNodeRet = graph.RemoveNode(destNode);
  FUSION_PASS_CHECK(
      removeNodeRet != SUCCESS,
      CUBE_INNER_ERR_REPORT(convNode->GetType().c_str(), "ConvNode[%s]: remove the destNode failed.", convNodeName.c_str()),
      return removeNodeRet);
  fusionNodes.push_back(convNode);
  return SUCCESS;
}

Status ConvFusionPassBase::GetConvFilterInputIndex(const ge::NodePtr& convNode, int& filterInputIdx) {
  for (ge::InDataAnchorPtr inDataAnchorPtr : convNode->GetAllInDataAnchors()) {
    if (inDataAnchorPtr == nullptr || inDataAnchorPtr->GetPeerOutAnchor() == nullptr ||
        inDataAnchorPtr->GetPeerOutAnchor()->GetOwnerNode() == nullptr) {
      continue;
    }
    ge::NodePtr nodePtr = inDataAnchorPtr->GetPeerOutAnchor()->GetOwnerNode();
    std::string type = ge::NodeUtils::GetInConstNodeTypeCrossSubgraph(nodePtr);
    if (type == CONSTANT || type == CONSTANTOP || nodePtr->GetType() == CONVBNFILTERHOST ||
        nodePtr->GetType() == GROUPPADDING || nodePtr->GetType() == CONCATHOSTOP) {
      if (inDataAnchorPtr->GetIdx() == 0) {
        continue;
      }
      filterInputIdx = inDataAnchorPtr->GetIdx();
      break;
    }
  }
  FUSION_PASS_CHECK(filterInputIdx < 0,
                    OP_LOGI(convNode->GetType().c_str(), "ConvNode[%s]: the conv node does not have a const input.",
                            convNode->GetName().c_str()),
                    return FAILED);
  OP_LOGI(convNode->GetType().c_str(), "ConvNode[%s]: the filter index of const input is [%d].",
          convNode->GetName().c_str(), filterInputIdx);
  return SUCCESS;
}

Status ConvFusionPassBase::GetAllConstInput(const ge::NodePtr& node, vector<ge::GeTensorDesc>& conv2dInputs,
                                            vector<string>& conv2dInputsName,
                                            vector<ge::InDataAnchorPtr>& conv2dInputAncors,
                                            vector<ge::GeTensorDesc>& constOutputs,
                                            vector<ge::OutDataAnchorPtr>& constOutputAncors) {
  for (ge::InDataAnchorPtr inDataAnchorPtr : node->GetAllInDataAnchors()) {
    if (inDataAnchorPtr == nullptr || inDataAnchorPtr->GetPeerOutAnchor() == nullptr ||
        inDataAnchorPtr->GetPeerOutAnchor()->GetOwnerNode() == nullptr) {
      continue;
    }
    ge::NodePtr nodePtr = inDataAnchorPtr->GetPeerOutAnchor()->GetOwnerNode();
    std::string type = ge::NodeUtils::GetInConstNodeTypeCrossSubgraph(nodePtr);
    bool checkConstNode = type == CONSTANT || type == CONSTANTOP || nodePtr->GetType() == CONVBNFILTERHOST ||
                          nodePtr->GetType() == CONVBNBIASHOST || nodePtr->GetType() == GROUPPADDING ||
                          nodePtr->GetType() == CONCATHOSTOP;
    if (checkConstNode) {
      if (inDataAnchorPtr->GetIdx() == 0) {
        continue;
      }
      conv2dInputs.push_back(node->GetOpDesc()->GetInputDesc(inDataAnchorPtr->GetIdx()));
      string inputName = node->GetOpDesc()->GetInputNameByIndex(inDataAnchorPtr->GetIdx());

      inputName = inputName.substr(inputName.find_last_of("_") + 1);
      if (inputName == "w") {
        inputName = "scale";
      }
      if (inputName == "b") {
        inputName = "bias";
      }
      if (node->GetType() == CONV2D || node->GetType() == DEPTHWISECONV2D || node->GetType() == CONV3D) {
        conv2dInputsName.push_back("Conv_" + inputName);
      } else {
        conv2dInputsName.push_back(node->GetOpDesc()->GetType() + "_" + inputName);
      }
      conv2dInputAncors.push_back(inDataAnchorPtr);
      constOutputs.push_back(nodePtr->GetOpDesc()->GetOutputDesc(0));
      constOutputAncors.push_back(inDataAnchorPtr->GetPeerOutAnchor());
      FUSION_PASS_CHECK(ge::GraphUtils::RemoveEdge(inDataAnchorPtr->GetPeerOutAnchor(), inDataAnchorPtr) != SUCCESS,
                        OP_LOGI(node->GetType().c_str(), "remove edge error"), return FAILED);
    }
  }
  return SUCCESS;
}

Status ConvFusionPassBase::AddBiasNode(ge::ComputeGraph& graph, ge::NodePtr& convNode) {
  // if the former conv has no bias, newBiasData = transBias
  ge::OpDescPtr constOpDesc = nullptr;
  FUSION_PASS_MAKE_SHARED((constOpDesc = std::make_shared<ge::OpDesc>(convNode->GetName() + "_" + "bias", CONSTANT)),
                          constOpDesc = nullptr;
                          return PARAM_INVALID);
  ge::GeTensorDesc constOutDesc;
  FUSION_PASS_CHECK(constOpDesc->AddOutputDesc(constOutDesc) != SUCCESS,
                    CUBE_INNER_ERR_REPORT("AddBiasNode", "AddOutputDesc failed!"), return FAILED);
  ge::NodePtr constNode = graph.AddNode(constOpDesc);
  ge::GeTensorPtr biasPtr = nullptr;
  FUSION_PASS_MAKE_SHARED((biasPtr = std::make_shared<ge::GeTensor>(constOutDesc, (uint8_t*)0, sizeof(float))),
                          biasPtr = nullptr;
                          return PARAM_INVALID);
  ge::GeShape biasShape({1});
  biasPtr->MutableTensorDesc().SetShape(biasShape);
  FUSION_PASS_CHECK(constNode == nullptr,
                    CUBE_INNER_ERR_REPORT("AddBiasNode", "constNode is nullptr"), return PARAM_INVALID);
  vector<ge::GeTensorPtr> weights;
  weights.push_back(biasPtr);
  ge::OpDescUtils::SetWeights(constNode, weights);
  // bias is the name of the third input of conv2d in IR conv2d.h
  Status res = convNode->AddLinkFrom("bias", constNode);
  FUSION_PASS_CHECK(res != SUCCESS,
                    CUBE_INNER_ERR_REPORT(convNode->GetType().c_str(),
                            "ConvNode[%s]: add edge between new const node and conv "
                            "node failed!",
                            convNode->GetName().c_str()),
                    return res);

  return SUCCESS;
}

Status ConvFusionPassBase::GetConvKernelIndex(ge::OpDescPtr convOpdesc, const ge::GeTensorDesc& constInputDesc,
                                              ge::Format& filterFormat, size_t& kernerlIndex) {
  filterFormat = constInputDesc.GetOriginFormat();
  if (filterFormat == ge::FORMAT_NCHW) {
    kernerlIndex = NCHW_DIM_N;
  } else if (filterFormat == ge::FORMAT_NHWC) {
    kernerlIndex = NHWC_DIM_N;
  } else if (filterFormat == ge::FORMAT_HWCN) {
    kernerlIndex = HWCN_DIM_N;
  } else if (filterFormat == ge::FORMAT_CHWN) {
    kernerlIndex = CHWN_DIM_N;
  } else if (filterFormat == ge::FORMAT_DHWCN) {
    kernerlIndex = DHWCN_DIM_N;
  } else {
    OP_LOGD(convOpdesc->GetType().c_str(), "ConvNode[%s]: the filter format [%d] is not supported.",
            convOpdesc->GetName().c_str(), filterFormat);
    return FAILED;
  }
  return SUCCESS;
}

Status ConvFusionPassBase::GetConvChannelIndex(ge::OpDescPtr convOpdesc, const ge::GeTensorDesc& constInputDesc,
                                               ge::Format& filterFormat, size_t& channelIndex) {
  filterFormat = constInputDesc.GetOriginFormat();
  if (filterFormat == ge::FORMAT_NCHW) {
    channelIndex = NCHW_DIM_C;
  } else if (filterFormat == ge::FORMAT_NHWC) {
    channelIndex = NHWC_DIM_C;
  } else if (filterFormat == ge::FORMAT_HWCN) {
    channelIndex = HWCN_DIM_C;
  } else if (filterFormat == ge::FORMAT_CHWN) {
    channelIndex = CHWN_DIM_C;
  } else if (filterFormat == ge::FORMAT_DHWCN) {
    channelIndex = DHWCN_DIM_C;
  } else {
    OP_LOGD(convOpdesc->GetType().c_str(), "ConvNode[%s]: the filter format [%d] is not supported.",
            convOpdesc->GetName().c_str(), filterFormat);
    return FAILED;
  }
  return SUCCESS;
}
}  // namespace fe
