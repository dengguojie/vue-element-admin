/**
 * Copyright 2021 Huawei Technologies Co., Ltd
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
 * \file transdata_fusion_pass.cpp
 * \brief diag transdata pass
 */
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
#include "fp16_t.hpp"
#include "graph_optimizer/graph_fusion/fusion_pass_manager/fusion_pass_registry.h"
#include "securec.h"
#include "pattern_fusion_util.h"
#include "transdata_fusion_pass.h"
#include "tbe_ops_pass_util.h"

using namespace std;
using namespace ge;

namespace fe {
static const string PATTERN_TRANSDATA = "TransData";
static const char* TRANSDATA = "TransData";

vector<FusionPattern*> TransDataPass::DefinePatterns() {
  vector<FusionPattern*> patterns;
  // define Fusion
  FusionPattern* pattern = new (std::nothrow) FusionPattern("TransDataPass");
  FUSION_PASS_CHECK(pattern == nullptr, OP_LOGE(FUSED_OP_TYPE.c_str(), "new a pattern object failed."),
                    return patterns);
  // define origin graph
  pattern->AddOpDesc(PATTERN_TRANSDATA, {TRANSDATA}).SetOutput(PATTERN_TRANSDATA);

  patterns.push_back(pattern);

  return patterns;
}

Status TransDataPass::Fusion(ge::ComputeGraph& graph, Mapping& mapping, vector<ge::NodePtr>& fusionNodes) {
  OP_LOGI(FUSED_OP_TYPE.c_str(), "enter into TransDataPass");
  // diag node
  ge::NodePtr transdataNode = GetNodeFromMapping(PATTERN_TRANSDATA, mapping);
  FUSION_PASS_CHECK(transdataNode == nullptr, OP_LOGE(FUSED_OP_TYPE.c_str(), "transdataNode is null, fusion failed."),
                    return PARAM_INVALID);

  // check input and output link relation
  FUSION_PASS_CHECK(transdataNode->GetInDataNodes().size() != 1,
                    OP_LOGI(FUSED_OP_TYPE.c_str(), "Input node of transdata node size is [%lu], which not equal to 1.",
                            transdataNode->GetInDataNodes().size()),
                    return NOT_CHANGED);
  FUSION_PASS_CHECK(transdataNode->GetOutDataNodes().size() != 1,
                    OP_LOGI(FUSED_OP_TYPE.c_str(), "Output node of transdata node size is [%d], which not equal to 1.",
                            transdataNode->GetOutDataNodes().size()),
                    return NOT_CHANGED);

  // get input and output shape and format
  Operator transdataOp = OpDescUtils::CreateOperatorFromNode(transdataNode);
  TensorDesc inputDesc = transdataOp.GetInputDesc("src");
  Format inputFormat = inputDesc.GetFormat();
  Shape inputShape = inputDesc.GetShape();
  DataType inputDtype = inputDesc.GetDataType();
  TensorDesc outputDesc = transdataOp.GetOutputDesc("dst");
  Format outputFormat = outputDesc.GetFormat();
  Shape outputShape = outputDesc.GetShape();

  // check dynamic shape
  FUSION_PASS_CHECK(IsUnknownShape(inputShape.GetDims()), OP_LOGI(FUSED_OP_TYPE.c_str(), "TransData is dynamic."),
                    return NOT_CHANGED);
  // check input dtype
  FUSION_PASS_CHECK(inputDtype != DT_FLOAT, OP_LOGI(FUSED_OP_TYPE.c_str(), "Input dtype must be float."),
                    return NOT_CHANGED);

  int32_t hDim = 1;
  int32_t wDim = 1;
  int32_t cDim = 1;
  if (inputFormat == FORMAT_NC1HWC0) {
    vector<int64_t> outDims = outputShape.GetDims();
    FUSION_PASS_CHECK(outDims.size() != 4, OP_LOGI(FUSED_OP_TYPE.c_str(), "the length of output shape must be 4."),
                      return NOT_CHANGED);
    if (outputFormat == FORMAT_NHWC) {
      hDim = outDims[1];
      wDim = outDims[2];
      cDim = outDims[3];
    } else if (outputFormat == FORMAT_NCHW) {
      hDim = outDims[2];
      wDim = outDims[3];
      cDim = outDims[1];
    } else {
      return NOT_CHANGED;
    }
  } else if (outputFormat == FORMAT_NC1HWC0) {
    vector<int64_t> inDims = inputShape.GetDims();
    FUSION_PASS_CHECK(inDims.size() != 4, OP_LOGI(FUSED_OP_TYPE.c_str(), "the length of input shape must be 4."),
                      return NOT_CHANGED);
    if (inputFormat == FORMAT_NHWC) {
      hDim = inDims[1];
      wDim = inDims[2];
      cDim = inDims[3];
    } else if (inputFormat == FORMAT_NCHW) {
      hDim = inDims[2];
      wDim = inDims[3];
      cDim = inDims[1];
    } else {
      return NOT_CHANGED;
    }
  } else {
    return NOT_CHANGED;
  }

  // check h and w and c dim
  if ((hDim != 1) || (wDim != 1) || (cDim % 16 != 0)) {
    OP_LOGI(FUSED_OP_TYPE.c_str(), "h and w and c dim not meet the relus, not changed.");
    return NOT_CHANGED;
  }

  // get input and output node
  auto inDataAnchor = transdataNode->GetInDataAnchor(0);
  auto inDataAnchorPeerOutAnchor = inDataAnchor->GetPeerOutAnchor();
  auto inputNode = inDataAnchorPeerOutAnchor->GetOwnerNode();
  auto outDataAnchor = transdataNode->GetOutDataAnchor(0);
  auto outDataAnchorPeerInDataAnchors = outDataAnchor->GetPeerInDataAnchors();
  // unlink
  for (auto inAnchor : transdataNode->GetAllInDataAnchors()) {
    if (inAnchor != nullptr) {
      inAnchor->UnlinkAll();
    }
  }
  if (transdataNode->GetInControlAnchor() != nullptr) {
    transdataNode->GetInControlAnchor()->UnlinkAll();
  }

  for (auto outAnchor : transdataNode->GetAllOutDataAnchors()) {
    if (outAnchor != nullptr) {
      outAnchor->UnlinkAll();
    }
  }
  if (transdataNode->GetOutControlAnchor() != nullptr) {
    transdataNode->GetOutControlAnchor()->UnlinkAll();
  }
  // add edge
  for (uint64_t i = 0; i < outDataAnchorPeerInDataAnchors.size(); i++) {
    auto peerInDataAnchor = outDataAnchorPeerInDataAnchors.at(i);
    auto outputNode = peerInDataAnchor->GetOwnerNode();
    FUSION_PASS_CHECK(SUCCESS != ge::GraphUtils::AddEdge(inDataAnchorPeerOutAnchor, peerInDataAnchor),
                      OP_LOGE(FUSED_OP_TYPE.c_str(), "add edge from src node[%s] to dst node[%s] failed.",
                              inputNode->GetName().c_str(), outputNode->GetName().c_str()),
                      return FAILED);
  }
  // remove node
  FUSION_PASS_CHECK(ge::GRAPH_SUCCESS != graph.RemoveNode(transdataNode),
                    OP_LOGE(FUSED_OP_TYPE.c_str(), "remove transdata node failed"), return FAILED);

  OP_LOGI(FUSED_OP_TYPE.c_str(), "TransDataPass success!!!!");

  return SUCCESS;
}
REGISTER_PASS("TransDataPass", SECOND_ROUND_BUILT_IN_GRAPH_PASS, TransDataPass);
}  // namespace fe
