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
 * \file max_pool3d_grad_grad_fusion_pass.cpp
 * \brief MaxPool3DGradGrad fusion pass(MaxPool3DGradGrad --> MaxPool3DGradGradD)
 */
#include "max_pool3d_grad_grad_fusion_pass.h"

#include <iostream>
#include <vector>
#include <string>
#include <map>

#include "graph/utils/op_desc_utils.h"
#include "graph/utils/graph_utils.h"
#include "graph/utils/node_utils.h"
#include "graph/utils/attr_utils.h"
#include "graph/debug/ge_attr_define.h"
#include "fp16_t.hpp"
#include "op_log.h"
#include "error_util.h"
#include "pattern_fusion_util.h"

#include "graph_optimizer/graph_fusion/fusion_pass_manager/fusion_pass_registry.h"
#include "securec.h"

using namespace std;
using namespace ge;

namespace fe {
static const float FLOAT_NUM_ZERO = 0;
static const int32_t INT_NUM_ZERO = 0;
static const uint16_t UINT_NUM_ZERO = 0;
static const string PATTERN_MAX_POOL3D_GRAD_GRAD = "MaxPool3DGradGrad";
static const std::string CONSTANTOP = "Constant";
static const char* MAX_POOL3D_GRAD_GRAD = "MaxPool3DGradGrad";
static const int32_t FP32_C0 = 16;
static const int32_t FP16_C0 = 16;

Status MaxPool3DGradGradAssitHelpFP32(int32_t D, int32_t H, int32_t W, int32_t n, float* output) {
  int maximum = D * H * W;
  int counter = 0;
  for (int32_t i = 0; i < n; i = i + FP32_C0) {
    int32_t base = i - i % FP32_C0;
    for (int32_t j = 0; j < FP32_C0; j++) {
      output[base + j] = maximum - counter;
    }
    counter++;
  }
  return SUCCESS;
}

Status MaxPool3DGradGradAssitHelpFP16(int32_t D, int32_t H, int32_t W, int32_t n, uint16_t* output) {
  int maximum = D * H * W;
  int counter = 0;
  for (int32_t i = 0; i < n; i = i + FP16_C0) {
    int32_t base = i - i % FP16_C0;
    for (int32_t j = 0; j < FP16_C0; j++) {
      output[base + j] = maximum - counter;
    }
    counter++;
  }
  return SUCCESS;
}

vector<FusionPattern*> MaxPool3DGradGradFusionPass::DefinePatterns() {
  vector<FusionPattern*> patterns;
  FusionPattern* pattern = new (std::nothrow) FusionPattern("MaxPool3DGradGradFusionPass");
  FUSION_PASS_CHECK(pattern == nullptr, VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "new a pattern object failed."),
                    return patterns);

  pattern->AddOpDesc(PATTERN_MAX_POOL3D_GRAD_GRAD, {MAX_POOL3D_GRAD_GRAD}).SetOutput(PATTERN_MAX_POOL3D_GRAD_GRAD);
  patterns.push_back(pattern);

  return patterns;
}

int MaxPool3DGradGradFusionPass::GetDHW(const std::vector<int32_t>& ksize, int32_t& D, int32_t& H, int32_t& W,
                                        Format format) {
  if (ksize.size() != 1 && ksize.size() != 3 && ksize.size() != 5) {
    VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "ksize.size() is %lu, should be 1 3 or 5", ksize.size());
    return -1;
  }

  if (ksize.size() == 1) {
    D = H = W = ksize[0];
  } else if (ksize.size() == 3) {
    D = ksize[0];
    H = ksize[1];
    W = ksize[2];
  } else {
    if (format == ge::FORMAT_NDHWC) {
      D = ksize[1];
      H = ksize[2];
      W = ksize[3];
    } else if (format == ge::FORMAT_NCDHW) {
      D = ksize[2];
      H = ksize[3];
      W = ksize[4];
    } else {
      VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "Unsupported shape for MaxPool3DGradGrad. format = %d", format);
      return -1;
    }
  }
  return 0;
}

Status MaxPool3DGradGradFusionPass::Fusion(ge::ComputeGraph& graph, Mapping& mapping,
                                           vector<ge::NodePtr>& fusionNodes) {
  int32_t D = 0;
  int32_t H = 0;
  int32_t W = 0;

  ge::NodePtr maxpool3dGradGradVNode = GetNodeFromMapping(PATTERN_MAX_POOL3D_GRAD_GRAD, mapping);

  FUSION_PASS_CHECK(maxpool3dGradGradVNode == nullptr,
                    VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "maxpool3dGradGradVNode is null, fusion failed."),
                    return PARAM_INVALID);

  ge::OpDescPtr maxpool3dGradGradDesc = maxpool3dGradGradVNode->GetOpDesc();
  FUSION_PASS_CHECK(maxpool3dGradGradDesc == nullptr,
                    VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "maxpool3dGradGradVNode's OpDesc is null, fusion failed."),
                    return PARAM_INVALID);

  ge::OpDescPtr fusionDesc = AttrUtils::CopyOpDesc(maxpool3dGradGradDesc);

  ge::InDataAnchorPtr maxpool3dGradGradAnchorPtr0 = maxpool3dGradGradVNode->GetInDataAnchor(0);
  ge::OutDataAnchorPtr constAnchorPtr0 = maxpool3dGradGradAnchorPtr0->GetPeerOutAnchor();
  ge::NodePtr constNode0 = constAnchorPtr0->GetOwnerNode();

  ge::GeTensorDesc diagpartInputTensor = constNode0->GetOpDesc()->GetOutputDesc(0);
  Format origFormat = diagpartInputTensor.GetOriginFormat();
  DataType origDataType = diagpartInputTensor.GetOriginDataType();
  DataType dataType = diagpartInputTensor.GetDataType();

  Operator maxpool3dGradGradOperator = ge::OpDescUtils::CreateOperatorFromNode(maxpool3dGradGradVNode);

  std::vector<int32_t> ksize;
  FUSION_PASS_CHECK(maxpool3dGradGradOperator.GetAttr("ksize", ksize) == ge::GRAPH_FAILED,
                    OP_LOGW(FUSED_OP_TYPE.c_str(), "get attr ksize failed."), return ge::NOT_CHANGED);

  FUSION_PASS_CHECK(GetDHW(ksize, D, H, W, origFormat) == -1,
                    OP_LOGW(FUSED_OP_TYPE.c_str(), "maxpool3dGradGradVNode's GetDHW failed, fusion failed."),
                    return NOT_CHANGED);

  vector<int64_t> assistShapeVec;
  assistShapeVec.push_back(1);  // N
  assistShapeVec.push_back(D);  // D
  assistShapeVec.push_back(1);  // C1
  assistShapeVec.push_back(H);  // H
  assistShapeVec.push_back(W);  // W

  ge::GeTensorPtr assitPtr = nullptr;
  ge::GeTensorDesc tensorDesc(GeShape(), ge::FORMAT_NDC1HWC0, ge::DT_FLOAT16);
  tensorDesc.SetFormat(ge::FORMAT_NDC1HWC0);
  tensorDesc.SetOriginFormat(ge::FORMAT_NDC1HWC0);
  tensorDesc.SetOriginDataType(origDataType);

  if (dataType == ge::DT_FLOAT) {
    assistShapeVec.push_back(FP32_C0);
    ge::GeShape assistShape(assistShapeVec);
    int64_t dimNums = 1 * D * 1 * H * W * FP32_C0;
    unique_ptr<float[]> inputAssit(new (std::nothrow) float[dimNums]());
    FUSION_PASS_CHECK(inputAssit.get() == nullptr, VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "inputAssit is NULL"),
                      return PARAM_INVALID);
    Status ret = NnSet(dimNums, FLOAT_NUM_ZERO, *reinterpret_cast<float*>(inputAssit.get()));
    FUSION_PASS_CHECK(ret != SUCCESS, OP_LOGW(FUSED_OP_TYPE.c_str(), "NnSet failed."), return NOT_CHANGED);
    ret = MaxPool3DGradGradAssitHelpFP32(D, H, W, dimNums, inputAssit.get());
    FUSION_PASS_CHECK(ret != SUCCESS, OP_LOGW(FUSED_OP_TYPE.c_str(), "AssitHelpFP32 failed."), return NOT_CHANGED);
    tensorDesc.SetShape(assistShape);
    tensorDesc.SetOriginShape(assistShape);
    tensorDesc.SetDataType(ge::DT_FLOAT);
    FUSION_PASS_MAKE_SHARED((assitPtr = std::make_shared<ge::GeTensor>(
                                 tensorDesc, reinterpret_cast<uint8_t*>(inputAssit.get()), dimNums * sizeof(float))),
                            assitPtr = nullptr;
                            return PARAM_INVALID);
  } else if (dataType == ge::DT_FLOAT16) {
    assistShapeVec.push_back(FP16_C0);
    ge::GeShape assistShape(assistShapeVec);
    int64_t dimNums = 1 * D * 1 * H * W * FP16_C0;
    unique_ptr<uint16_t[]> inputAssit(new (std::nothrow) uint16_t[dimNums]());
    FUSION_PASS_CHECK(inputAssit.get() == nullptr, VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "inputAssit is NULL"),
                      return PARAM_INVALID);
    Status ret = NnSet(dimNums, UINT_NUM_ZERO, *reinterpret_cast<uint16_t*>(inputAssit.get()));
    FUSION_PASS_CHECK(ret != SUCCESS, OP_LOGW(FUSED_OP_TYPE.c_str(), "NnSet failed."), return NOT_CHANGED);
    ret = MaxPool3DGradGradAssitHelpFP16(D, H, W, dimNums, inputAssit.get());
    FUSION_PASS_CHECK(ret != SUCCESS, OP_LOGW(FUSED_OP_TYPE.c_str(), "AssitHelpFP16 failed."), return NOT_CHANGED);
    tensorDesc.SetShape(assistShape);
    tensorDesc.SetOriginShape(assistShape);
    tensorDesc.SetDataType(ge::DT_FLOAT16);
    FUSION_PASS_MAKE_SHARED((assitPtr = std::make_shared<ge::GeTensor>(
                                 tensorDesc, reinterpret_cast<uint8_t*>(inputAssit.get()), dimNums * sizeof(uint16_t))),
                            assitPtr = nullptr;
                            return PARAM_INVALID);
  }

  fusionDesc->AddInputDesc(tensorDesc);
  fusionDesc->SetType("MaxPool3DGradGradD");
  if (!CheckOpSupported(fusionDesc)){
    assitPtr = nullptr;
    OP_LOGI(FUSED_OP_TYPE.c_str(), "Op DiagPart Not Supported.");
    return NOT_CHANGED;
  }
  vector<ge::GeTensorPtr> weights = {assitPtr};
  ge::OpDescUtils::SetWeights(maxpool3dGradGradVNode, weights);
  auto constInputNodes = OpDescUtils::GetConstInputs(maxpool3dGradGradVNode);
  if (constInputNodes.size() == 0){
    assitPtr = nullptr;
    VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "constInputNodes is empty, fusion failed.");
    return PARAM_INVALID;
  }
  NodePtr constInput = constInputNodes[0];
  if (constInput == nullptr){
    assitPtr = nullptr;
    VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "constInput is null, fusion failed.");
    return PARAM_INVALID;
  }
  constInput->GetOpDesc()->SetType(CONSTANTOP);
  maxpool3dGradGradDesc->SetType("MaxPool3DGradGradD");

  return SUCCESS;
}

REGISTER_PASS("MaxPool3DGradGradFusionPass", BUILT_IN_GRAPH_PASS, MaxPool3DGradGradFusionPass);
}  // namespace fe
