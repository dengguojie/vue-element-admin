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
 * \file tril_fusion_pass.cpp
 * \brief
 */
#include "tril_fusion_pass.h"

#include <iostream>
#include <map>
#include <string>
#include <vector>

#include "fp16_t.hpp"
#include "graph/debug/ge_attr_define.h"
#include "graph/utils/attr_utils.h"
#include "graph/utils/graph_utils.h"
#include "graph/utils/node_utils.h"
#include "graph/utils/op_desc_utils.h"
#include "graph_optimizer/graph_fusion/fusion_pass_manager/fusion_pass_registry.h"
#include "op_log.h"
#include "pattern_fusion_util.h"
#include "securec.h"

using namespace std;
using namespace ge;

namespace fe {
// the name of tf op
static const float FLOAT_NUM_ZERO = 0;
static const int32_t INT_NUM_ZERO = 0;
static const uint16_t UINT_NUM_ZERO = 0;
static const int8_t INT8_NUM_ZERO = 0;
static const uint8_t UINT8_NUM_ZERO = 0;
static const string PATTERN_TRIL = "Tril";
static const std::string CONSTANTOP = "Constant";
static const char* TRIL = "Tril";

// dim dim1 dim2
template <typename Dtype>
Status AssitHelp(const int32_t n, const int32_t m, const int32_t x, Dtype& output1, int64_t diagonal) {
  Dtype* output = &output1;
  int32_t matrix_num = n / (m * x);
  int32_t index = 0;
  for (int32_t i = 0; i < matrix_num; ++i) {
    for (int32_t j = 0; j < x; ++j) {
      for (int32_t k = 0; k < m; ++k) {
        if (j > k - diagonal) {
          index = i * x * m + j * m + k;
          output[index] = 1;
        }
      }
    }
  }
  return SUCCESS;
}

template <typename Dtype>
Status AssitHelp1(const int32_t n, const int32_t m, const int32_t x, Dtype& output1, int64_t diagonal) {
  Dtype* output = &output1;
  fp16_t t;
  t.val = 0;
  int32_t xx = 1;
  t = xx;
  int32_t matrix_num = n / (m * x);
  int32_t index = 0;
  for (int32_t i = 0; i < matrix_num; ++i) {
    for (int32_t j = 0; j < x; ++j) {
      for (int32_t k = 0; k < m; ++k) {
        if (j > k - diagonal) {
          index = i * x * m + j * m + k;
          output[index] = t.val;
        }
      }
    }
  }
  return SUCCESS;
}

vector<FusionPattern*> TrilFusionPass::DefinePatterns() {
  vector<FusionPattern*> patterns;

  // tril->mul
  // define TrilFusion
  FusionPattern* pattern = new (std::nothrow) FusionPattern("TrilFusionPass");
  FUSION_PASS_CHECK(pattern == nullptr, OP_LOGE(FUSED_OP_TYPE.c_str(), "new a pattern object failed."),
                    return patterns);

  // define origin graph
  pattern->AddOpDesc(PATTERN_TRIL, {TRIL}).SetOutput(PATTERN_TRIL);

  patterns.push_back(pattern);

  return patterns;
}

Status TrilFusionPass::Fusion(ge::ComputeGraph& graph, Mapping& mapping, vector<ge::NodePtr>& fusionNodes) {
    // tril node
  bool is_dynamic_shape = false;
  ge::NodePtr trilVNode = GetNodeFromMapping(PATTERN_TRIL, mapping);
  FUSION_PASS_CHECK(trilVNode == nullptr,
                    OP_LOGE(FUSED_OP_TYPE.c_str(),
                            "trilVNode is null, "
                            "fusion failed."),
                    return PARAM_INVALID);

  // input of tril
  ge::OpDescPtr trilDesc = trilVNode->GetOpDesc();
  FUSION_PASS_CHECK(trilDesc == nullptr,
                    OP_LOGE(FUSED_OP_TYPE.c_str(),
                            "trilVNode's OpDesc is "
                            "null, fusion failed."),
                    return PARAM_INVALID);
  // get fuzz build attr
  ge::AttrUtils::GetBool(trilVNode->GetOpDesc(), ge::ATTR_NAME_FUZZ_BUILD, is_dynamic_shape);
  if (is_dynamic_shape) {
    OP_LOGD(FUSED_OP_TYPE.c_str(), "is dynamic shape.");
    return NOT_CHANGED;
  } else {
    OP_LOGD(FUSED_OP_TYPE.c_str(), "is not dynamic shape.");
  }

  ge::OpDescPtr fusionDesc = AttrUtils::CopyOpDesc(trilDesc);

  // get the input desc of the entrance of tril node to differentiate between
  // const and var
  ge::GeTensorDesc trilInputTensor = trilVNode->GetOpDesc()->GetInputDesc(0);
  OP_LOGD(FUSED_OP_TYPE.c_str(), "get input success.");

  // get the diagonal attr
  int64_t diagonal;
  AttrUtils::GetInt(trilDesc, "diagonal", diagonal);
  diagonal = diagonal + 1;
  OP_LOGD(FUSED_OP_TYPE.c_str(), "get attr diagonal success");
  // get the shape info
  ge::GeShape trilInputShape = trilInputTensor.GetShape();

  // get the data type
  DataType dataType = trilInputTensor.GetDataType();

  // multiples of dims
  int64_t dimNums = 1;
  int64_t dimsInput = trilInputShape.GetDimNum() - 1;
  int64_t dimsInput1 = trilInputShape.GetDimNum() - 2;
  for (size_t j = 0; j < trilInputShape.GetDimNum(); ++j) {
    dimNums = trilInputShape.GetDim(j) * dimNums;
  }

  // get the last dims of input shape
  int64_t dimNums1 = trilInputShape.GetDim(dimsInput);
  int64_t dimNums2 = trilInputShape.GetDim(dimsInput1);

  vector<int64_t> dimInfo = trilInputShape.GetDims();

  Format assitMatrixFormat = trilInputTensor.GetFormat();

  ge::GeTensorPtr assitPtr = nullptr;
  ge::GeTensorDesc tensorDesc(GeShape(), ge::FORMAT_ND, ge::DT_FLOAT);
  if (dataType == ge::DT_FLOAT) {
    unique_ptr<float[]> inputAssit(new (std::nothrow) float[dimNums]());
    FUSION_PASS_CHECK(inputAssit.get() == nullptr, OP_LOGE(FUSED_OP_TYPE.c_str(), "inputAssit is NULL"),
                      return PARAM_INVALID);
    Status ret = NnSet(dimNums, FLOAT_NUM_ZERO, *reinterpret_cast<float*>(inputAssit.get()));
    FUSION_PASS_CHECK(ret != SUCCESS, OP_LOGW(FUSED_OP_TYPE.c_str(), "NnSet failed."), return NOT_CHANGED);

    ret = AssitHelp(dimNums, dimNums1, dimNums2, *inputAssit.get(), diagonal);
    FUSION_PASS_CHECK(ret != SUCCESS, OP_LOGW(FUSED_OP_TYPE.c_str(), "AssitHelp failed."), return NOT_CHANGED);

    // define the shape of auxiliary matrix
    ge::GeShape assitShape = trilInputShape;
    tensorDesc.SetShape(assitShape);
    tensorDesc.SetOriginShape(assitShape);
    tensorDesc.SetFormat(assitMatrixFormat);
    tensorDesc.SetOriginFormat(assitMatrixFormat);
    tensorDesc.SetDataType(ge::DT_FLOAT);
    tensorDesc.SetOriginDataType(ge::DT_FLOAT);
    FUSION_PASS_MAKE_SHARED(
      (assitPtr = std::make_shared<ge::GeTensor>(tensorDesc, reinterpret_cast<uint8_t*>(inputAssit.get()),
                                                 dimNums * sizeof(float))),
      assitPtr = nullptr;
      return PARAM_INVALID);
  } else if (dataType == ge::DT_INT32) {
    unique_ptr<int32_t[]> inputAssit(new (std::nothrow) int32_t[dimNums]());
    FUSION_PASS_CHECK(inputAssit.get() == nullptr, OP_LOGE(FUSED_OP_TYPE.c_str(), "inputAssit is NULL"),
                      return PARAM_INVALID);

    Status ret = NnSet(dimNums, INT_NUM_ZERO, *reinterpret_cast<int32_t*>(inputAssit.get()));
    FUSION_PASS_CHECK(ret != SUCCESS, OP_LOGW(FUSED_OP_TYPE.c_str(), "NnSet failed."), return NOT_CHANGED);

    ret = AssitHelp(dimNums, dimNums1, dimNums2, *inputAssit.get(), diagonal);
    FUSION_PASS_CHECK(ret != SUCCESS, OP_LOGW(FUSED_OP_TYPE.c_str(), "AssitHelp failed."), return NOT_CHANGED);

    // define the shape of auxiliary matrix
    ge::GeShape assitShape = trilInputShape;
    tensorDesc.SetShape(assitShape);
    tensorDesc.SetOriginShape(assitShape);
    tensorDesc.SetFormat(assitMatrixFormat);
    tensorDesc.SetOriginFormat(assitMatrixFormat);
    tensorDesc.SetDataType(ge::DT_INT32);
    tensorDesc.SetOriginDataType(ge::DT_INT32);
    FUSION_PASS_MAKE_SHARED(
      (assitPtr = std::make_shared<ge::GeTensor>(tensorDesc, reinterpret_cast<uint8_t*>(inputAssit.get()),
                                                 dimNums * sizeof(int32_t))),
      assitPtr = nullptr;
      return PARAM_INVALID);
  } else if (dataType == ge::DT_FLOAT16) {
    unique_ptr<uint16_t[]> inputAssit(new (std::nothrow) uint16_t[dimNums]());
    FUSION_PASS_CHECK(inputAssit.get() == nullptr, OP_LOGE(FUSED_OP_TYPE.c_str(), "inputAssit is NULL"),
                      return PARAM_INVALID);
    Status ret = NnSet(dimNums, UINT_NUM_ZERO, *reinterpret_cast<uint16_t*>(inputAssit.get()));
    FUSION_PASS_CHECK(ret != SUCCESS, OP_LOGW(FUSED_OP_TYPE.c_str(), "NnSet failed."), return NOT_CHANGED);

    ret = AssitHelp1(dimNums, dimNums1, dimNums2, *inputAssit.get(), diagonal);
    FUSION_PASS_CHECK(ret != SUCCESS, OP_LOGW(FUSED_OP_TYPE.c_str(), "AssitHelp failed."), return NOT_CHANGED);

    // define the shape of auxiliary matrix
    ge::GeShape assitShape = trilInputShape;
    tensorDesc.SetShape(assitShape);
    tensorDesc.SetOriginShape(assitShape);
    tensorDesc.SetFormat(assitMatrixFormat);
    tensorDesc.SetOriginFormat(assitMatrixFormat);
    tensorDesc.SetDataType(ge::DT_FLOAT16);
    tensorDesc.SetOriginDataType(ge::DT_FLOAT16);
    FUSION_PASS_MAKE_SHARED(
      (assitPtr = std::make_shared<ge::GeTensor>(tensorDesc, reinterpret_cast<uint8_t*>(inputAssit.get()),
                                                 dimNums * sizeof(uint16_t))),
      assitPtr = nullptr;
      return PARAM_INVALID);
  } else if (dataType == ge::DT_INT8) {
    unique_ptr<int8_t[]> inputAssit(new (std::nothrow) int8_t[dimNums]());
    FUSION_PASS_CHECK(inputAssit.get() == nullptr, OP_LOGE(FUSED_OP_TYPE.c_str(), "inputAssit is NULL"),
                      return PARAM_INVALID);

    Status ret = NnSet(dimNums, INT8_NUM_ZERO, *reinterpret_cast<int8_t*>(inputAssit.get()));
    FUSION_PASS_CHECK(ret != SUCCESS, OP_LOGW(FUSED_OP_TYPE.c_str(), "NnSet failed."), return NOT_CHANGED);

    ret = AssitHelp(dimNums, dimNums1, dimNums2, *inputAssit.get(), diagonal);
    FUSION_PASS_CHECK(ret != SUCCESS, OP_LOGW(FUSED_OP_TYPE.c_str(), "AssitHelp failed."), return NOT_CHANGED);

    // define the shape of auxiliary matrix
    ge::GeShape assitShape = trilInputShape;
    tensorDesc.SetShape(assitShape);
    tensorDesc.SetOriginShape(assitShape);
    tensorDesc.SetFormat(assitMatrixFormat);
    tensorDesc.SetOriginFormat(assitMatrixFormat);
    tensorDesc.SetDataType(ge::DT_INT8);
    tensorDesc.SetOriginDataType(ge::DT_INT8);
    FUSION_PASS_MAKE_SHARED(
      (assitPtr = std::make_shared<ge::GeTensor>(tensorDesc, reinterpret_cast<uint8_t*>(inputAssit.get()),
                                                 dimNums * sizeof(int8_t))),
      assitPtr = nullptr;
      return PARAM_INVALID);
  } else if (dataType == ge::DT_UINT8) {
    unique_ptr<uint8_t[]> inputAssit(new (std::nothrow) uint8_t[dimNums]());
    FUSION_PASS_CHECK(inputAssit.get() == nullptr, OP_LOGE(FUSED_OP_TYPE.c_str(), "inputAssit is NULL"),
                      return PARAM_INVALID);

    Status ret = NnSet(dimNums, UINT8_NUM_ZERO, *reinterpret_cast<uint8_t*>(inputAssit.get()));
    FUSION_PASS_CHECK(ret != SUCCESS, OP_LOGW(FUSED_OP_TYPE.c_str(), "NnSet failed."), return NOT_CHANGED);

    ret = AssitHelp(dimNums, dimNums1, dimNums2, *inputAssit.get(), diagonal);
    FUSION_PASS_CHECK(ret != SUCCESS, OP_LOGW(FUSED_OP_TYPE.c_str(), "AssitHelp failed."), return NOT_CHANGED);

    // define the shape of auxiliary matrix
    ge::GeShape assitShape = trilInputShape;
    tensorDesc.SetShape(assitShape);
    tensorDesc.SetOriginShape(assitShape);
    tensorDesc.SetFormat(assitMatrixFormat);
    tensorDesc.SetOriginFormat(assitMatrixFormat);
    tensorDesc.SetDataType(ge::DT_UINT8);
    tensorDesc.SetOriginDataType(ge::DT_UINT8);
    FUSION_PASS_MAKE_SHARED(
      (assitPtr = std::make_shared<ge::GeTensor>(tensorDesc, reinterpret_cast<uint8_t*>(inputAssit.get()),
                                                 dimNums * sizeof(uint8_t))),
      assitPtr = nullptr;
      return PARAM_INVALID);
  } else {
    return NOT_CHANGED;
  }
  // check op support
  vector<ge::GeTensorPtr> weights = {assitPtr};
  ge::OpDescUtils::SetWeights(trilVNode, weights);
  auto constInputNodes = OpDescUtils::GetConstInputs(trilVNode);
  NodePtr constInput = constInputNodes[0];
  constInput->GetOpDesc()->SetType(CONSTANTOP);
  trilDesc->SetType("Mul");

  return SUCCESS;
}

REGISTER_PASS("TrilFusionPass", BUILT_IN_GRAPH_PASS, TrilFusionPass);
}  // namespace fe
