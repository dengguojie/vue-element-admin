/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2020-2021. All rights reserved.
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
 * \file prelu_abs_fusion_pass.cpp
 * \brief LayerNormGrad fusion pass
 *   (LayerNormGrad --> LayerNormXBackprop & LayerNormBetaGammaBackprop)
 */
#include "adaptive_max_pool2d_fusion_pass.h"

#include <iostream>
#include <vector>
#include <string>
#include <map>

#include "op_log.h"
#include "error_util.h"
#include "pattern_fusion_util.h"
#include "graph/utils/op_desc_utils.h"
#include "graph/utils/graph_utils.h"
#include "graph/utils/node_utils.h"
#include "graph/utils/attr_utils.h"
#include "graph/debug/ge_attr_define.h"
#include "graph_optimizer/graph_fusion/fusion_pass_manager/fusion_pass_registry.h"

using namespace ge;
using namespace std;

namespace fe {

static const char* FUSED_NODE = "AdaptiveMaxPool2d";
static const string POOLING_NODE = "Pooling";
static const std::string PATTERN_FUSEDNODE = "AdaptiveMaxPool2d";

vector<FusionPattern*> AdaptiveMaxPool2dFusionPass::DefinePatterns() {
  vector<FusionPattern*> patterns;

  FusionPattern* pattern = new (std::nothrow) FusionPattern("AdaptiveMaxPool2dFusionPass");
  FUSION_PASS_CHECK(pattern == nullptr, VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(),
                   "new a pattern object failed."),
                    return patterns);

  pattern->AddOpDesc(PATTERN_FUSEDNODE, {FUSED_NODE}).SetOutput(PATTERN_FUSEDNODE);

  patterns.push_back(pattern);

  return patterns;
}

std::vector<int> compute_kernel(int input_size, int output_size){
  int padding_size = 0;
  int kernel_size = 0;
  int stride_size = 0;
  int res0 = 0;

  int ceil_mode = 1;
  
  if (input_size % output_size == 0){
    kernel_size = stride_size = input_size / output_size;
    padding_size = 0;
    return std::vector<int>{0, kernel_size, stride_size, padding_size, ceil_mode};
  } else {
    for(kernel_size=(input_size / output_size + 1);kernel_size<=(input_size / output_size + 2);kernel_size++){
      for(stride_size=(input_size / output_size);stride_size<=(input_size / output_size + 1);stride_size++){
        res0 = (input_size + 2 * padding_size - kernel_size) / stride_size + 1;
        if (res0 == output_size){
          return std::vector<int>{1, kernel_size, stride_size, padding_size, ceil_mode};
        }
      }
    }
    while (padding_size >= 0) {
      for(kernel_size=padding_size + 1;kernel_size<=input_size + 2 * padding_size;kernel_size++){
        for(stride_size=1;stride_size<=kernel_size;stride_size++){
          res0 = (input_size + 2 * padding_size - kernel_size) / stride_size + 1;
          if (res0 == output_size){
            return std::vector<int>{2, kernel_size, stride_size, padding_size, ceil_mode};
          }
        }
      }
      padding_size++;
      if (padding_size >= 10){
        break;
      }
    }
  return std::vector<int>{-1, -1, -1, -1, ceil_mode};
  }
}

Status AdaptiveMaxPool2dFusionPass::Fusion(ge::ComputeGraph& graph, Mapping& mapping, vector<ge::NodePtr>& newNodes) {
  OP_LOGD(FUSED_OP_TYPE.c_str(), "Enter graph fusion AdaptiveMaxPool2dFusionPass!");

  ge::NodePtr fusedNode = GetNodeFromMapping(PATTERN_FUSEDNODE, mapping);
  FUSION_PASS_CHECK(fusedNode == nullptr, VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(),
                    "fusedNode is null, fusion failed."),
                    return PARAM_INVALID);
  ge::OpDescPtr fusedDesc = fusedNode->GetOpDesc();
  FUSION_PASS_CHECK(fusedDesc == nullptr, VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(),
                   "fusedNode's OpDesc is null, fusion failed."),
                    return PARAM_INVALID);
  Operator op = ge::OpDescUtils::CreateOperatorFromNode(fusedNode);
  std::vector<int> output_sizeList;
  if (op.GetAttr("output_size", output_sizeList) != ge::GRAPH_SUCCESS){
    VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "GetOpAttr output_size failed");
    return FAILED;
  }
  ge::GeTensorDesc outputTensorDesc = fusedDesc->GetOutputDesc(0);
  ge::GeShape output_shape = outputTensorDesc.GetShape();
  ge::GeTensorDesc inputTensorDesc = fusedDesc->GetInputDesc(0);
  Format input_format = inputTensorDesc.GetFormat();
  string data_format;
  ge::GeShape shape = inputTensorDesc.GetShape();
  int in_size_h = 0;
  int in_size_w = 0;
  int out_size_h = 0;
  int out_size_w = 0;
  if (input_format == ge::FORMAT_NHWC) {
    in_size_h = shape.GetDim(1);
    in_size_w = shape.GetDim(2);
    data_format = "NHWC";
  } else if (input_format == ge::FORMAT_NCHW) {
    in_size_h = shape.GetDim(2);
    in_size_w = shape.GetDim(3);
    data_format = "NCHW";
  } else if (input_format == ge::FORMAT_NCHW){
    in_size_h = shape.GetDim(1);
    in_size_w = shape.GetDim(2);
    data_format = "ND";
  } else {
    VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "Not support this format!");
    return FAILED;
  }
  out_size_h = output_sizeList[0];
  out_size_w = output_sizeList[1];
  if (in_size_h % out_size_h != 0 || in_size_w % out_size_w != 0){
    OP_LOGI(FUSED_OP_TYPE.c_str(), "Not Fusion, because input_size can not be divided by output_size!");
    return NOT_CHANGED;
  }
  std::vector<int> flag_h;
  flag_h = compute_kernel(in_size_h, out_size_h);
  std::vector<int> flag_w;
  flag_w = compute_kernel(in_size_w, out_size_w);

  if (flag_h[0] == -1 || flag_w[0] == -1){
    VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "Not support this scene!");
    return FAILED;
  }

  std::vector<int> windowValue;
  windowValue.push_back(flag_h[1]);
  windowValue.push_back(flag_w[1]);

  std::vector<int> strideValue;
  strideValue.push_back(flag_h[2]);
  strideValue.push_back(flag_w[2]);

  std::vector<int> padValue;
  padValue.push_back(flag_h[3]);
  padValue.push_back(flag_h[3]);
  padValue.push_back(flag_w[3]);
  padValue.push_back(flag_w[3]);

  int64_t ceilmodeValue = 1;
  int64_t modeValue = 0;
  std::vector<int> dilationValue = {1, 1, 1, 1};

  std::string poolingNodeName = fusedNode->GetName() + "_" + "pooling";
  std::shared_ptr<ge::OpDesc> poolingOpdesc = std::make_shared<ge::OpDesc>(poolingNodeName, POOLING_NODE);
  FUSION_PASS_CHECK(poolingOpdesc == nullptr, VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(),
                   "poolingOpdesc is null, fusion failed."),
                    return PARAM_INVALID);
  FUSION_PASS_CHECK(poolingOpdesc->AddInputDesc(0, inputTensorDesc) != SUCCESS,
                    VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "add poolingNode input desc failed."),
                    return FAILED);
  FUSION_PASS_CHECK(poolingOpdesc->AddOutputDesc(outputTensorDesc) != SUCCESS,
                    VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "add poolingNode output desc failed."),
                    return FAILED);
  FUSION_PASS_CHECK(!ge::AttrUtils::SetInt(poolingOpdesc, "mode", modeValue),
                    VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(),
                    "Set attr %s to node %s error", "mode", poolingOpdesc->GetName().c_str()),
                    return FAILED);
  FUSION_PASS_CHECK(!ge::AttrUtils::SetListInt(poolingOpdesc, "window", windowValue),
                    VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(),
                    "Set attr %s to node %s error", "window", poolingOpdesc->GetName().c_str()),
                    return FAILED);
  FUSION_PASS_CHECK(!ge::AttrUtils::SetListInt(poolingOpdesc, "stride", strideValue),
                    VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(),
                    "Set attr %s to node %s error", "stride", poolingOpdesc->GetName().c_str()),
                    return FAILED);
  FUSION_PASS_CHECK(!ge::AttrUtils::SetListInt(poolingOpdesc, "pad", padValue),
                    VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(),
                    "Set attr %s to node %s error", "pad", poolingOpdesc->GetName().c_str()),
                    return FAILED);
  FUSION_PASS_CHECK(!ge::AttrUtils::SetInt(poolingOpdesc, "ceil_mode", ceilmodeValue),
                    VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(),
                    "Set attr %s to node %s error", "ceil_mode", poolingOpdesc->GetName().c_str()),
                    return FAILED);
  FUSION_PASS_CHECK(!ge::AttrUtils::SetStr(poolingOpdesc, "data_format", data_format),
                    VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(),
                    "Set attr %s to node %s error", "data_format", poolingOpdesc->GetName().c_str()),
                    return FAILED);
  FUSION_PASS_CHECK(!ge::AttrUtils::SetBool(poolingOpdesc, "global_pooling", false),
                    VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(),
                    "Set attr %s to node %s error", "global_pooling", poolingOpdesc->GetName().c_str()),
                    return FAILED);
  FUSION_PASS_CHECK(!ge::AttrUtils::SetListInt(poolingOpdesc, "dilation", dilationValue),
                    VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(),
                    "Set attr %s to node %s error", "dilation", poolingOpdesc->GetName().c_str()),
                    return FAILED);

  ge::NodePtr poolingNode = graph.AddNode(poolingOpdesc);
  poolingNode->GetOpDesc()->SetType(POOLING_NODE);
  newNodes.push_back(poolingNode);

  FUSION_PASS_CHECK(ge::GraphUtils::AddEdge(fusedNode->GetInDataAnchor(0)->GetPeerOutAnchor(),
                                            poolingNode->GetInDataAnchor(0)) != SUCCESS,
                    VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(),
                    "Add edge between fusedNode and poolingNode failed."), return FAILED);

  for (auto &inDataAnchor : fusedNode->GetOutDataAnchor(0)->GetPeerInDataAnchors()) {
    FUSION_PASS_CHECK(ge::GraphUtils::RemoveEdge(fusedNode->GetOutDataAnchor(0), inDataAnchor) != SUCCESS,
                      VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(),
                      "Remove out data edge failed."), return FAILED);
    FUSION_PASS_CHECK(ge::GraphUtils::AddEdge(poolingNode->GetOutDataAnchor(0), inDataAnchor) != SUCCESS,
                      VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(),
                      "Add out data edge failed."), return FAILED);
  }

  FUSION_PASS_CHECK(graph.RemoveNode(fusedNode) != SUCCESS, VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(),
                   "Remove fusedNode failed."),
                    return FAILED);

  return SUCCESS;
}

REGISTER_PASS("AdaptiveMaxPool2dFusionPass", BUILT_IN_GRAPH_PASS, AdaptiveMaxPool2dFusionPass);

}  // namespace fe
