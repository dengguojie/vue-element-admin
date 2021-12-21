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
 * \file diag_part_fusion_pass.cpp
 * \brief diag fusion pass(DiagPart --> DiagPartD)
 */
#include "diag_part_fusion_pass.h"

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
// the name of tf op
static const float FLOAT_NUM_ZERO = 0;
static const int32_t INT_NUM_ZERO = 0;
static const uint16_t UINT_NUM_ZERO = 0;
static const string PATTERN_DIAGPART = "DiagPart";
static const std::string CONSTANTOP = "Constant";
static const char* DIAGPART = "DiagPart";

template <typename Dtype>
Status AssitHelp(const int32_t n, Dtype& output1) {
  Dtype* output = &output1;
  for (int32_t i = 0; i < n; ++i) {
    output[(1 + n) * i] = 1;
  }
  return SUCCESS;
}

template <typename Dtype>
Status AssitHelpFP16(const int32_t n, Dtype& output1) {
  Dtype* output = &output1;
  for (int32_t i = 0; i < n; ++i) {
    fp16_t t;
    t.val = 0;
    int32_t xx = 1;
    t = xx;
    output[(1 + n) * i] = t.val;
  }
  return SUCCESS;
}

vector<FusionPattern*> DiagPartFusionPass::DefinePatterns() {
  vector<FusionPattern*> patterns;

  // diag_part->diag_part_d
  // define DiagPartFusion
  FusionPattern* pattern = new (std::nothrow) FusionPattern("DiagPartFusionPass");
  FUSION_PASS_CHECK(pattern == nullptr, VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "new a pattern object failed."),
                    return patterns);

  // define origin graph
  pattern->AddOpDesc(PATTERN_DIAGPART, {DIAGPART}).SetOutput(PATTERN_DIAGPART);

  patterns.push_back(pattern);

  return patterns;
}

Status DiagPartFusionPass::Fusion(ge::ComputeGraph& graph, Mapping& mapping, vector<ge::NodePtr>& fusionNodes) {
  // diag_part node
  ge::NodePtr diagpartVNode = GetNodeFromMapping(PATTERN_DIAGPART, mapping);
  FUSION_PASS_CHECK(diagpartVNode == nullptr,
                    VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(),
                            "diagpartVNode is null, "
                            "fusion failed."),
                    return PARAM_INVALID);

  // input of diag_part
  ge::OpDescPtr diagpartDesc = diagpartVNode->GetOpDesc();
  FUSION_PASS_CHECK(diagpartDesc == nullptr,
                    VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(),
                            "diagpartVNode's OpDesc is "
                            "null, fusion failed."),
                    return PARAM_INVALID);
  ge::OpDescPtr fusionDesc = AttrUtils::CopyOpDesc(diagpartDesc);

  ge::GeTensorDesc diagpartInputTensor = diagpartVNode->GetOpDesc()->GetInputDesc(0);

  // get the shape info
  ge::GeShape diagpartInputShape = diagpartInputTensor.GetShape();

  // get the data type
  DataType dataType = diagpartInputTensor.GetDataType();

  // dynamic diag_part fusion
  auto x_input_shape = diagpartInputShape.GetDims();
  auto x_dim = x_input_shape.size();
  for (size_t i = 0; i < x_dim; i++) {
    FUSION_PASS_CHECK((x_input_shape[i] < 0),
                      OP_LOGI(FUSED_OP_TYPE, "Node:dynamic diag_part does not need fusion."),
                      return NOT_CHANGED);
  }

  // multipies of half dims
  int64_t dimNums = 1;
  for (size_t j = 0; j < diagpartInputShape.GetDimNum() / 2; ++j) {
    if (PatternFusionUtil::IsUnknownShape(diagpartInputShape.GetDim(j))) {
      VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "DiagPartFusionPass cannot be applied for unknown shape.");
      return NOT_CHANGED;
    }
    dimNums = diagpartInputShape.GetDim(j) * dimNums;
  }

  // GESHAPE->vector
  vector<int64_t> dimInfo = diagpartInputShape.GetDims();

  // Format
  Format assitMatrixFormat = diagpartInputTensor.GetFormat();

  ge::GeTensorPtr assitPtr = nullptr;
  ge::GeTensorDesc tensorDesc(GeShape(), ge::FORMAT_NCHW, ge::DT_FLOAT);
  if (dataType == ge::DT_FLOAT) {
    unique_ptr<float[]> inputAssit(new (std::nothrow) float[dimNums * dimNums]());
    FUSION_PASS_CHECK(inputAssit.get() == nullptr, VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "inputAssit is NULL"),
                      return PARAM_INVALID);
    Status ret = NnSet(dimNums * dimNums, FLOAT_NUM_ZERO, *reinterpret_cast<float*>(inputAssit.get()));
    FUSION_PASS_CHECK(ret != SUCCESS, VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "NnSet failed."), return ret);

    ret = AssitHelp(dimNums, *inputAssit.get());
    FUSION_PASS_CHECK(ret != SUCCESS, VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "AssitHelp failed."), return ret);

    // define the shape of auxiliary matrix
    ge::GeShape assitShape = diagpartInputShape;
    tensorDesc.SetShape(assitShape);
    tensorDesc.SetFormat(assitMatrixFormat);
    tensorDesc.SetDataType(ge::DT_FLOAT);
    FUSION_PASS_MAKE_SHARED(
        (assitPtr = std::make_shared<ge::GeTensor>(tensorDesc, reinterpret_cast<uint8_t*>(inputAssit.get()),
                                                   dimNums * dimNums * sizeof(float))),
        assitPtr = nullptr;
        return PARAM_INVALID);
  } else if (dataType == ge::DT_INT32) {
    unique_ptr<int32_t[]> inputAssit(new (std::nothrow) int32_t[dimNums * dimNums]());
    FUSION_PASS_CHECK(inputAssit.get() == nullptr, VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "inputAssit is NULL"),
                      return PARAM_INVALID);

    Status ret = NnSet(dimNums * dimNums, INT_NUM_ZERO, *reinterpret_cast<int32_t*>(inputAssit.get()));
    FUSION_PASS_CHECK(ret != SUCCESS, VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "NnSet failed."), return ret);

    ret = AssitHelp(dimNums, *inputAssit.get());
    FUSION_PASS_CHECK(ret != SUCCESS, VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "AssitHelp failed."), return ret);

    // define the shape of auxiliary matrix
    ge::GeShape assitShape = diagpartInputShape;
    tensorDesc.SetShape(assitShape);
    tensorDesc.SetFormat(assitMatrixFormat);
    tensorDesc.SetDataType(ge::DT_INT32);
    FUSION_PASS_MAKE_SHARED(
        (assitPtr = std::make_shared<ge::GeTensor>(tensorDesc, reinterpret_cast<uint8_t*>(inputAssit.get()),
                                                   dimNums * dimNums * sizeof(int32_t))),
        assitPtr = nullptr;
        return PARAM_INVALID);
  } else if (dataType == ge::DT_FLOAT16) {
    unique_ptr<uint16_t[]> inputAssit(new (std::nothrow) uint16_t[dimNums * dimNums]());
    FUSION_PASS_CHECK(inputAssit.get() == nullptr, VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "inputAssit is NULL"),
                      return PARAM_INVALID);

    Status ret = NnSet(dimNums * dimNums, UINT_NUM_ZERO, *reinterpret_cast<uint16_t*>(inputAssit.get()));
    FUSION_PASS_CHECK(ret != SUCCESS, VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "NnSet failed."), return ret);

    ret = AssitHelpFP16(dimNums, *inputAssit.get());
    FUSION_PASS_CHECK(ret != SUCCESS, VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "AssitHelpFP16 failed."), return ret);

    // define the shape of auxiliary matrix
    ge::GeShape assitShape = diagpartInputShape;
    tensorDesc.SetShape(assitShape);
    tensorDesc.SetFormat(assitMatrixFormat);
    tensorDesc.SetDataType(ge::DT_FLOAT16);
    FUSION_PASS_MAKE_SHARED(
        (assitPtr = std::make_shared<ge::GeTensor>(tensorDesc, reinterpret_cast<uint8_t*>(inputAssit.get()),
                                                   dimNums * dimNums * sizeof(uint16_t))),
        assitPtr = nullptr;
        return PARAM_INVALID);
  }
  fusionDesc->AddInputDesc(tensorDesc);
  fusionDesc->SetType("DiagPartD");
  // check op support
  FUSION_PASS_CHECK(!CheckOpSupported(fusionDesc), OP_LOGI(FUSED_OP_TYPE.c_str(), "Op DiagPart Not Supported."),
                    return NOT_CHANGED);
  vector<ge::GeTensorPtr> weights = {assitPtr};
  ge::OpDescUtils::SetWeights(diagpartVNode, weights);
  auto constInputNodes = OpDescUtils::GetConstInputs(diagpartVNode);
  NodePtr constInput = constInputNodes[0];
  constInput->GetOpDesc()->SetType(CONSTANTOP);
  diagpartDesc->SetType("DiagPartD");

  return SUCCESS;
}

REGISTER_PASS("DiagPartFusionPass", BUILT_IN_GRAPH_PASS, DiagPartFusionPass);
}  // namespace fe
