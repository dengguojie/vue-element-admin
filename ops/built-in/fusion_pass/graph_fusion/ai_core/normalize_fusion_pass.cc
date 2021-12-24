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
 * \file normalize_fusion_pass.cpp
 * \brief LayerNormGrad fusion pass
 *   (LayerNormGrad --> LayerNormXBackprop & LayerNormBetaGammaBackprop)
 */
#include "normalize_fusion_pass.h"

#include <iostream>
#include <vector>
#include <string>
#include <map>

#include "graph/utils/op_desc_utils.h"
#include "graph/utils/graph_utils.h"
#include "graph/utils/node_utils.h"
#include "graph/utils/attr_utils.h"
#include "graph/debug/ge_attr_define.h"
#include "graph_optimizer/graph_fusion/fusion_pass_manager/fusion_pass_registry.h"
#include "op_log.h"
#include "error_util.h"
#include "pattern_fusion_util.h"

using namespace ge;
namespace fe {

static const char* FUSED_NODE = "Normalize";

static const std::string PATTERN_FUSEDNODE = "Normalize";

vector<FusionPattern*> NormalizeFusionPass::DefinePatterns() {
  vector<FusionPattern*> patterns;

  FusionPattern* pattern = new (std::nothrow) FusionPattern("NormalizeFusionPass");
  FUSION_PASS_CHECK(pattern == nullptr,
                    VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "new a pattern object failed."),
                    return patterns);

  pattern->AddOpDesc(PATTERN_FUSEDNODE, {FUSED_NODE}).SetOutput(PATTERN_FUSEDNODE);

  patterns.push_back(pattern);

  return patterns;
}

Status NormalizeFusionPass::Fusion(ge::ComputeGraph& graph, Mapping& mapping, vector<ge::NodePtr>& newNodes) {
  // get the NodePtr of Normalize
  ge::NodePtr fusedNode = GetNodeFromMapping(PATTERN_FUSEDNODE, mapping);
  FUSION_PASS_CHECK(fusedNode == nullptr,
                    VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "fusedNode is null, fusion failed."),
                    return PARAM_INVALID);
  // get the OpDescPtr of Normalize
  ge::OpDescPtr fusedDesc = fusedNode->GetOpDesc();
  FUSION_PASS_CHECK(fusedDesc == nullptr,
                    VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "fusedNode's OpDesc is null, fusion failed."),
                    return PARAM_INVALID);
  // clone the OpDescPtr for NormalizeSum, including the input/output/attr
  ge::OpDescPtr normalizeSumDesc = AttrUtils::CloneOpDesc(fusedDesc);
  FUSION_PASS_CHECK(
      normalizeSumDesc == nullptr,
      VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(),
        "Node:%s's OpDesc is null, fusion failed.", fusedNode->GetName().c_str()),
      return PARAM_INVALID);
  // clone the OpDescPtr for NormalizeScale, including the input/output/attr
  ge::OpDescPtr normalizeScaleDesc = AttrUtils::CloneOpDesc(fusedDesc);
  FUSION_PASS_CHECK(
      normalizeScaleDesc == nullptr,
      VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(),
        "Node:%s's OpDesc is null, fusion failed.", fusedNode->GetName().c_str()),
      return PARAM_INVALID);
  normalizeSumDesc->SetName(fusedDesc->GetName() + "/NormalizeSum");
  normalizeScaleDesc->SetName(fusedDesc->GetName() + "/NormalizeScale");
  normalizeSumDesc->SetType("NormalizeSum");
  normalizeScaleDesc->SetType("NormalizeScale");

  // remove the input 1 from NormalizeSum, NormalizeSum only has 1 input
  OpDescUtils::ClearInputDesc(normalizeSumDesc, 1);
  // remove the attribute channel_shared,eps from NormalizeSum
  normalizeSumDesc->DelAttr("channel_shared");
  normalizeSumDesc->DelAttr("eps");

  bool across_spatial = true;

  Operator op = ge::OpDescUtils::CreateOperatorFromNode(fusedNode);
  op.GetAttr("across_spatial", across_spatial);

  // set dtype of NormalizeSum's output 0 as float16 when original is int8
  ge::GeTensorDesc outputTensorDesc = normalizeSumDesc->GetOutputDesc(0);
  ge::Format data_format = outputTensorDesc.GetFormat();
  ge::GeShape input_shape = outputTensorDesc.GetShape();
  std::vector<int64_t> input_shape_vector = input_shape.GetDims();
  std::vector<int64_t> output_shape_vector;
  unsigned int shapeSize = input_shape_vector.size();
 
  FUSION_PASS_CHECK(
    shapeSize != 4 && shapeSize != 2,
    VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(),
      "Node %s's Only Support 4 dim and 2 dim", fusedNode->GetName().c_str()),
    return PARAM_INVALID);

  if (across_spatial) {
    output_shape_vector.push_back(input_shape_vector[0]);
    output_shape_vector.push_back(1);
    output_shape_vector.push_back(1);
    output_shape_vector.push_back(1);
  } else {
    unsigned int outH = 1;
    unsigned int outW = 1;
    if (data_format == ge::FORMAT_NCHW) {
      if (shapeSize != 2) {
        outH = input_shape_vector[2];
        outW = input_shape_vector[3];
      }
      output_shape_vector.push_back(input_shape_vector[0]);
      output_shape_vector.push_back(1);
      output_shape_vector.push_back(outH);
      output_shape_vector.push_back(outW);
    } else if (data_format == ge::FORMAT_NHWC) {
      if (shapeSize != 2) {
        outH = input_shape_vector[1];
        outW = input_shape_vector[2];
      }
      output_shape_vector.push_back(input_shape_vector[0]);
      output_shape_vector.push_back(outH);
      output_shape_vector.push_back(outW);
      output_shape_vector.push_back(1);
    }
  }
  outputTensorDesc.SetShape(ge::GeShape(output_shape_vector));
  if (outputTensorDesc.GetDataType() == ge::DT_INT8) {
    outputTensorDesc.SetDataType(ge::DT_FLOAT16);
    outputTensorDesc.SetOriginDataType(ge::DT_INT8);
  }
  normalizeSumDesc->UpdateOutputDesc(0, outputTensorDesc);

  // add input 2 for NormalizeScale, clone it from NormalizeSum's output 0, NormalizeScale has 3 inputs
  normalizeScaleDesc->AddInputDesc("x3", normalizeSumDesc->GetOutputDesc(0));

  // add NormalizeSum and NormalizeScale to the graph
  ge::NodePtr normalizeSumNode = graph.AddNode(normalizeSumDesc);
  ge::NodePtr normalizeScaleNode = graph.AddNode(normalizeScaleDesc);
  FUSION_PASS_CHECK(
      normalizeSumNode == nullptr,
      VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(),
        "fusionNode:%s is null, fusion failed.", normalizeSumDesc->GetName().c_str()),
      return PARAM_INVALID);
  FUSION_PASS_CHECK(
      normalizeScaleNode == nullptr,
      VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(),
        "fusionNode:%s is null, fusion failed.", normalizeScaleDesc->GetName().c_str()),
      return PARAM_INVALID);

  // connect the input 0 of Normalize to input 0 of NormalizeSum
  FUSION_PASS_CHECK(
      SUCCESS != ge::GraphUtils::AddEdge(fusedNode->GetInDataAnchor(0)->GetPeerOutAnchor(),
                                         normalizeSumNode->GetInDataAnchor(0)),
      VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(),
        "Add edge from fused node:%s's input[%d] to fusion node:%s's input[%d] failed.",
              fusedNode->GetName().c_str(), 0, normalizeSumNode->GetName().c_str(), 0),
      return FAILED);
  OP_LOGD(FUSED_OP_TYPE.c_str(), "Add edge from fused node:%s's input[%d] to fusion node:%s's input[%d].",
          fusedNode->GetName().c_str(), 0, normalizeSumNode->GetName().c_str(), 0);

  // connect the input 0/1 of Normalize to input 0/1 of NormalizeScale
  for (unsigned int i = 0; i < fusedNode->GetAllInDataAnchors().size(); i++) {
    FUSION_PASS_CHECK(
        SUCCESS != ge::GraphUtils::AddEdge(fusedNode->GetInDataAnchor(i)->GetPeerOutAnchor(),
                                           normalizeScaleNode->GetInDataAnchor(i)),
        VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(),
          "Add edge from fused node:%s's input[%d] to fusion node:%s's input[%d] failed.",
                fusedNode->GetName().c_str(), i, normalizeScaleNode->GetName().c_str(), i),
        return FAILED);
    OP_LOGD(FUSED_OP_TYPE.c_str(), "Add edge from fused node:%s's input[%d] to fusion node:%s's input[%d].",
            fusedNode->GetName().c_str(), i, normalizeScaleNode->GetName().c_str(), i);
  }

  // connect the output 0 of NormalizeSum to input 2 of NormalizeScale
  FUSION_PASS_CHECK(
      SUCCESS != ge::GraphUtils::AddEdge(normalizeSumNode->GetOutDataAnchor(0), normalizeScaleNode->GetInDataAnchor(2)),
      VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(),
        "Add edge from fused node:%s's output[%d] to fusion node:%s's input[%d] failed.",
              normalizeSumNode->GetName().c_str(), 0, normalizeScaleNode->GetName().c_str(), 2),
      return FAILED);
  OP_LOGD(FUSED_OP_TYPE.c_str(), "Add edge from fused node:%s's output[%d] to fusion node:%s's input[%d].",
          normalizeSumNode->GetName().c_str(), 0, normalizeScaleNode->GetName().c_str(), 2);

  // connect the output 0 of NormalizeScale to output 0 of Normalize
  if (fusedNode->GetOutDataAnchor(0)->GetPeerInDataAnchors().size() > 0) {
    OP_LOGI(FUSED_OP_TYPE.c_str(), "The size of normalizeScaleNode is [%d].",
            fusedNode->GetOutDataAnchor(0)->GetPeerInDataAnchors().size());
    for (InDataAnchorPtr inAnchorPtr : fusedNode->GetOutDataAnchor(0)->GetPeerInDataAnchors()) {
      inAnchorPtr->UnlinkAll();
      FUSION_PASS_CHECK(SUCCESS != ge::GraphUtils::AddEdge(normalizeScaleNode->GetOutDataAnchor(0), inAnchorPtr),
                        VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(),
                                "Add edge from fused node:%s's output[0] to fusion node:%s's output[0] failed.",
                                normalizeScaleNode->GetName().c_str(), fusedNode->GetName().c_str()),
                        return FAILED);
      OP_LOGD(FUSED_OP_TYPE.c_str(), "Add edge from fused node:%s's output[0] to fusion node:%s's output[0].",
              normalizeScaleNode->GetName().c_str(), fusedNode->GetName().c_str());
    }
  }

  // unlink all control input of Normalize
  if (fusedNode->GetInControlAnchor() != nullptr) {
    // connect the control input of Normalize to control input of NormalizeSum
    for (unsigned int i = 0; i < fusedNode->GetInControlAnchor()->GetPeerOutControlAnchors().size(); i++) {
      FUSION_PASS_CHECK(
          SUCCESS != ge::GraphUtils::AddEdge(fusedNode->GetInControlAnchor()->GetPeerOutControlAnchors().at(i),
                                             normalizeSumNode->GetInControlAnchor()),
          VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(),
                  "Add edge from fused node:%s's control index[%d] to fusion node:%s's control index failed.",
                  fusedNode->GetName().c_str(), i, normalizeSumNode->GetName().c_str()),
          return FAILED);
      OP_LOGD(FUSED_OP_TYPE.c_str(),
              "Add edge from fused node:%s's control index[%d] to fusion node:%s's control index.",
              fusedNode->GetName().c_str(), i, normalizeSumNode->GetName().c_str());
    }
    fusedNode->GetInControlAnchor()->UnlinkAll();
  }

  // unlink all control output of Normalize
  if (fusedNode->GetOutControlAnchor() != nullptr) {
    // connect the control output of NormalizeScale to control output of Normalize
    for (unsigned int i = 0; i < fusedNode->GetOutControlAnchor()->GetPeerInControlAnchors().size(); i++) {
      FUSION_PASS_CHECK(
          SUCCESS != ge::GraphUtils::AddEdge(normalizeScaleNode->GetOutControlAnchor(),
                                             fusedNode->GetOutControlAnchor()->GetPeerInControlAnchors().at(i)),
          VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(),
                  "Add edge from fused node:%s's control index[%d] to fusion node:%s's control index failed.",
                  fusedNode->GetName().c_str(), i, normalizeScaleNode->GetName().c_str()),
          return FAILED);
      OP_LOGD(FUSED_OP_TYPE.c_str(),
              "Add edge from fused node:%s's control index[%d] to fusion node:%s's control index.",
              fusedNode->GetName().c_str(), i, normalizeScaleNode->GetName().c_str());
    }
    for (auto inControlAnchor : fusedNode->GetOutControlAnchor()->GetPeerInControlAnchors()) {
      FUSION_PASS_CHECK(SUCCESS != ge::GraphUtils::RemoveEdge(fusedNode->GetOutControlAnchor(), inControlAnchor),
                        VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(),
                          "Remove edge from fused node:%s's output control failed.", fusedNode->GetName().c_str()),
                        return FAILED);
      OP_LOGD(FUSED_OP_TYPE.c_str(), "Remove edge from fused node:%s's output control index.",
              fusedNode->GetName().c_str());
    }
  }

  // unlink all input of Normalize
  for (auto inAnchor : fusedNode->GetAllInDataAnchors()) {
    if (inAnchor != nullptr) {
      inAnchor->UnlinkAll();
    }
  }

  // remove Normalize from graph
  FUSION_PASS_CHECK(ge::GRAPH_SUCCESS != graph.RemoveNode(fusedNode),
                    VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(),
                      "remove fusedNode node[%s] failed", fusedNode->GetName().c_str()),
                    return FAILED);

  newNodes.push_back(normalizeSumNode);
  newNodes.push_back(normalizeScaleNode);
  return SUCCESS;
}

REGISTER_PASS("NormalizeFusionPass", BUILT_IN_GRAPH_PASS, NormalizeFusionPass);
}  // namespace fe
