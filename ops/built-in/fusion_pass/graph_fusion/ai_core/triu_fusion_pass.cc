/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2021-2021. All rights reserved.
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
 * \file triu_fusion_pass.cpp
 * \brief
 */
#include "triu_fusion_pass.h"

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
static const string PATTERN_TRIU = "Triu";
static const std::string CONSTANTOP = "Constant";
static const char *TRIU = "Triu";

// dim dim1 dim2
template <typename Dtype>
Status AssitHelp(const int32_t n, const int32_t m, const int32_t x, Dtype &output1, int64_t diagonal) {
  Dtype *output = &output1;
  int32_t z = m * x;
  int32_t y = n / z;
  int32_t dim12_index = 0;
  if (m > x) {
    for (int32_t i = 0; i < y; ++i) {
      for (int32_t j = 0; j < x; ++j) {
        for (int32_t k = 0; k < (m - j - diagonal); ++k) {
          dim12_index = (1 + m) * j + diagonal + k;
          if (dim12_index < 0 || dim12_index >= z) {
            continue;
          }
          output[(1 + m) * j + m * x * i + k + diagonal] = 1;
        }
      }
    }
  } else {
    for (int32_t i = 0; i < y; ++i) {
      for (int32_t j = 0; j < (m - diagonal); ++j) {
        for (int32_t k = 0; k < (m - j - diagonal); ++k) {
          dim12_index = (1 + m) * j + diagonal + k;
          if (dim12_index < 0 || dim12_index >= z) {
            continue;
          }
          output[(1 + m) * j + m * x * i + k + diagonal] = 1;
        }
      }
    }
  }

  return SUCCESS;
}

template <typename Dtype>
Status AssitHelp1(const int32_t n, const int32_t m, const int32_t x, Dtype &output1, int64_t diagonal) {
  Dtype *output = &output1;
  fp16_t t;
  t.val = 0;
  int32_t xx = 1;
  t = xx;
  int32_t z = m * x;
  int32_t y = n / z;
  int32_t dim12_index = 0;
  if (m > x) {
    for (int32_t i = 0; i < y; ++i) {
      for (int32_t j = 0; j < x; ++j) {
        for (int32_t k = 0; k < (m - j - diagonal); ++k) {
          dim12_index = (1 + m) * j + diagonal + k;
          if (dim12_index < 0 || dim12_index >= z) {
            continue;
          }
          output[(1 + m) * j + m * x * i + k + diagonal] = t.val;
        }
      }
    }
  } else {
    for (int32_t i = 0; i < y; ++i) {
      for (int32_t j = 0; j < (m - diagonal); ++j) {
        for (int32_t k = 0; k < (m - j - diagonal); ++k) {
          dim12_index = (1 + m) * j + diagonal + k;
          if (dim12_index < 0 || dim12_index >= z) {
            continue;
          }
          output[(1 + m) * j + m * x * i + k + diagonal] = t.val;
        }
      }
    }
  }

  return SUCCESS;
}

vector<FusionPattern *> TriuFusionPass::DefinePatterns() {
  vector<FusionPattern *> patterns;

  // triu->mul
  // define TriuFusion
  FusionPattern *pattern = new (std::nothrow) FusionPattern("TriuFusionPass");
  FUSION_PASS_CHECK(pattern == nullptr, OP_LOGE(FUSED_OP_TYPE.c_str(), "new a pattern object failed."),
                    return patterns);

  // define origin graph
  pattern->AddOpDesc(PATTERN_TRIU, {TRIU}).SetOutput(PATTERN_TRIU);

  patterns.push_back(pattern);

  return patterns;
}

Status TriuFusionPass::Fusion(ge::ComputeGraph &graph, Mapping &mapping, vector<ge::NodePtr> &fusionNodes) {
  bool is_dynamic_shape = false;
  OP_LOGD(FUSED_OP_TYPE.c_str(), "is_dynamic_shape default: false.");
  // triu node
  ge::NodePtr triuVNode = GetNodeFromMapping(PATTERN_TRIU, mapping);
  FUSION_PASS_CHECK(triuVNode == nullptr,
                    OP_LOGE(FUSED_OP_TYPE.c_str(),
                            "triuVNode is null, "
                            "fusion failed."),
                    return PARAM_INVALID);

  // input of triu
  ge::OpDescPtr triuDesc = triuVNode->GetOpDesc();
  FUSION_PASS_CHECK(triuDesc == nullptr,
                    OP_LOGE(FUSED_OP_TYPE.c_str(),
                            "triuVNode's OpDesc is "
                            "null, fusion failed."),
                    return PARAM_INVALID);

  // get fuzz build attr
  ge::AttrUtils::GetBool(triuVNode->GetOpDesc(), ge::ATTR_NAME_FUZZ_BUILD, is_dynamic_shape);
  if (is_dynamic_shape) {
      OP_LOGD(FUSED_OP_TYPE.c_str(), "is dynamic shape.");
      return NOT_CHANGED;
  } else {
      OP_LOGD(FUSED_OP_TYPE.c_str(), "is not dynamic shape.");
  }

  ge::OpDescPtr fusionDesc = AttrUtils::CopyOpDesc(triuDesc);
  // get the input desc of the entrance of triu node to differentiate between
  // const and var
  ge::GeTensorDesc triuInputTensor = triuVNode->GetOpDesc()->GetInputDesc(0);
  OP_LOGD(FUSED_OP_TYPE.c_str(), "get input success.");

  // get the diagonal attr
  int64_t diagonal;
  AttrUtils::GetInt(triuDesc, "diagonal", diagonal);
  OP_LOGD(FUSED_OP_TYPE.c_str(), "attr diagonal default: 0.");

  // get the shape info
  ge::GeShape triuInputShape = triuInputTensor.GetShape();

  // get the data type
  DataType dataType = triuInputTensor.GetDataType();

  // multiples of dims
  int64_t dimNums = 1;
  int64_t first_offset = 1;
  int64_t second_offset = 2;
  int64_t dimsInput = triuInputShape.GetDimNum() - first_offset;
  int64_t dimsInput1 = triuInputShape.GetDimNum() - second_offset;
  for (size_t j = 0; j < triuInputShape.GetDimNum(); ++j) {
    if (PatternFusionUtil::IsUnknownShape(triuInputShape.GetDim(j))) {
      OP_LOGE(FUSED_OP_TYPE.c_str(), "TriuFusionPass cannot be applied for unknown shape.");
      return NOT_CHANGED;
    }
    dimNums = triuInputShape.GetDim(j) * dimNums;
  }

  // get the last dims of input shape
  int64_t dimNums1 = triuInputShape.GetDim(dimsInput);
  int64_t dimNums2 = triuInputShape.GetDim(dimsInput1);
  if (PatternFusionUtil::IsUnknownShape(dimNums1) || PatternFusionUtil::IsUnknownShape(dimNums2)) {
    OP_LOGE(FUSED_OP_TYPE.c_str(), "TriuFusionPass cannot be applied for unknown shape.");
    return NOT_CHANGED;
  }
  vector<int64_t> dimInfo = triuInputShape.GetDims();
  Format assitMatrixFormat = triuInputTensor.GetFormat();

  ge::GeTensorPtr assitPtr = nullptr;
  ge::GeTensorDesc tensorDesc(GeShape(), ge::FORMAT_ND, ge::DT_FLOAT);
  if (dataType == ge::DT_FLOAT) {
    unique_ptr<float[]> inputAssit(new (std::nothrow) float[dimNums]());
    FUSION_PASS_CHECK(inputAssit.get() == nullptr, OP_LOGE(FUSED_OP_TYPE.c_str(), "inputAssit is NULL"),
                      return PARAM_INVALID);
    Status ret = NnSet(dimNums, FLOAT_NUM_ZERO, *reinterpret_cast<float *>(inputAssit.get()));
    FUSION_PASS_CHECK(ret != SUCCESS, OP_LOGW(FUSED_OP_TYPE.c_str(), "NnSet failed."), return NOT_CHANGED);
    ret = AssitHelp(dimNums, dimNums1, dimNums2, *inputAssit.get(), diagonal);
    FUSION_PASS_CHECK(ret != SUCCESS, OP_LOGW(FUSED_OP_TYPE.c_str(), "AssitHelp failed."), return NOT_CHANGED);

    // define the shape of auxiliary matrix
    ge::GeShape assitShape = triuInputShape;
    tensorDesc.SetShape(assitShape);
    tensorDesc.SetOriginShape(assitShape);
    tensorDesc.SetFormat(assitMatrixFormat);
    tensorDesc.SetOriginFormat(assitMatrixFormat);
    tensorDesc.SetDataType(ge::DT_FLOAT);
    tensorDesc.SetOriginDataType(ge::DT_FLOAT);
    FUSION_PASS_MAKE_SHARED(
      (assitPtr = std::make_shared<ge::GeTensor>(tensorDesc, reinterpret_cast<uint8_t *>(inputAssit.get()),
                                                 dimNums * sizeof(float))),
      assitPtr = nullptr;
      return PARAM_INVALID);
  } else if (dataType == ge::DT_INT32) {
    unique_ptr<int32_t[]> inputAssit(new (std::nothrow) int32_t[dimNums]());
    FUSION_PASS_CHECK(inputAssit.get() == nullptr, OP_LOGE(FUSED_OP_TYPE.c_str(), "inputAssit is NULL"),
                      return PARAM_INVALID);

    Status ret = NnSet(dimNums, INT_NUM_ZERO, *reinterpret_cast<int32_t *>(inputAssit.get()));
    FUSION_PASS_CHECK(ret != SUCCESS, OP_LOGW(FUSED_OP_TYPE.c_str(), "NnSet failed."), return NOT_CHANGED);
    ret = AssitHelp(dimNums, dimNums1, dimNums2, *inputAssit.get(), diagonal);
    FUSION_PASS_CHECK(ret != SUCCESS, OP_LOGW(FUSED_OP_TYPE.c_str(), "AssitHelp failed."), return NOT_CHANGED);
    // define the shape of auxiliary matrix
    ge::GeShape assitShape = triuInputShape;
    tensorDesc.SetShape(assitShape);
    tensorDesc.SetOriginShape(assitShape);
    tensorDesc.SetFormat(assitMatrixFormat);
    tensorDesc.SetOriginFormat(assitMatrixFormat);
    tensorDesc.SetDataType(ge::DT_INT32);
    tensorDesc.SetOriginDataType(ge::DT_INT32);
    FUSION_PASS_MAKE_SHARED(
      (assitPtr = std::make_shared<ge::GeTensor>(tensorDesc, reinterpret_cast<uint8_t *>(inputAssit.get()),
                                                 dimNums * sizeof(int32_t))),
      assitPtr = nullptr;
      return PARAM_INVALID);
  } else if (dataType == ge::DT_FLOAT16) {
    unique_ptr<uint16_t[]> inputAssit(new (std::nothrow) uint16_t[dimNums]());
    FUSION_PASS_CHECK(inputAssit.get() == nullptr, OP_LOGE(FUSED_OP_TYPE.c_str(), "inputAssit is NULL"),
                      return PARAM_INVALID);
    Status ret = NnSet(dimNums, UINT_NUM_ZERO, *reinterpret_cast<uint16_t *>(inputAssit.get()));
    FUSION_PASS_CHECK(ret != SUCCESS, OP_LOGW(FUSED_OP_TYPE.c_str(), "NnSet failed."), return NOT_CHANGED);

    ret = AssitHelp1(dimNums, dimNums1, dimNums2, *inputAssit.get(), diagonal);
    FUSION_PASS_CHECK(ret != SUCCESS, OP_LOGW(FUSED_OP_TYPE.c_str(), "AssitHelp failed."), return NOT_CHANGED);

    // define the shape of auxiliary matrix
    ge::GeShape assitShape = triuInputShape;
    tensorDesc.SetShape(assitShape);
    tensorDesc.SetOriginShape(assitShape);
    tensorDesc.SetFormat(assitMatrixFormat);
    tensorDesc.SetOriginFormat(assitMatrixFormat);
    tensorDesc.SetDataType(ge::DT_FLOAT16);
    tensorDesc.SetOriginDataType(ge::DT_FLOAT16);
    FUSION_PASS_MAKE_SHARED(
      (assitPtr = std::make_shared<ge::GeTensor>(tensorDesc, reinterpret_cast<uint8_t *>(inputAssit.get()),
                                                 dimNums * sizeof(uint16_t))),
      assitPtr = nullptr;
      return PARAM_INVALID);
  } else if (dataType == ge::DT_INT8) {
    unique_ptr<int8_t[]> inputAssit(new (std::nothrow) int8_t[dimNums]());
    FUSION_PASS_CHECK(inputAssit.get() == nullptr, OP_LOGE(FUSED_OP_TYPE.c_str(), "inputAssit is NULL"),
                      return PARAM_INVALID);

    Status ret = NnSet(dimNums, INT8_NUM_ZERO, *reinterpret_cast<int8_t *>(inputAssit.get()));
    FUSION_PASS_CHECK(ret != SUCCESS, OP_LOGW(FUSED_OP_TYPE.c_str(), "NnSet failed."), return NOT_CHANGED);

    ret = AssitHelp(dimNums, dimNums1, dimNums2, *inputAssit.get(), diagonal);
    FUSION_PASS_CHECK(ret != SUCCESS, OP_LOGW(FUSED_OP_TYPE.c_str(), "AssitHelp failed."), return NOT_CHANGED);

    // define the shape of auxiliary matrix
    ge::GeShape assitShape = triuInputShape;
    tensorDesc.SetShape(assitShape);
    tensorDesc.SetOriginShape(assitShape);
    tensorDesc.SetFormat(assitMatrixFormat);
    tensorDesc.SetOriginFormat(assitMatrixFormat);
    tensorDesc.SetDataType(ge::DT_INT8);
    tensorDesc.SetOriginDataType(ge::DT_INT8);
    FUSION_PASS_MAKE_SHARED(
      (assitPtr = std::make_shared<ge::GeTensor>(tensorDesc, reinterpret_cast<uint8_t *>(inputAssit.get()),
                                                 dimNums * sizeof(int8_t))),
      assitPtr = nullptr;
      return PARAM_INVALID);
  } else if (dataType == ge::DT_UINT8) {
    unique_ptr<uint8_t[]> inputAssit(new (std::nothrow) uint8_t[dimNums]());
    FUSION_PASS_CHECK(inputAssit.get() == nullptr, OP_LOGE(FUSED_OP_TYPE.c_str(), "inputAssit is NULL"),
                      return PARAM_INVALID);

    Status ret = NnSet(dimNums, UINT8_NUM_ZERO, *reinterpret_cast<uint8_t *>(inputAssit.get()));
    FUSION_PASS_CHECK(ret != SUCCESS, OP_LOGW(FUSED_OP_TYPE.c_str(), "NnSet failed."), return NOT_CHANGED);

    ret = AssitHelp(dimNums, dimNums1, dimNums2, *inputAssit.get(), diagonal);
    FUSION_PASS_CHECK(ret != SUCCESS, OP_LOGW(FUSED_OP_TYPE.c_str(), "AssitHelp failed."), return NOT_CHANGED);

    // define the shape of auxiliary matrix
    ge::GeShape assitShape = triuInputShape;
    tensorDesc.SetShape(assitShape);
    tensorDesc.SetOriginShape(assitShape);
    tensorDesc.SetFormat(assitMatrixFormat);
    tensorDesc.SetOriginFormat(assitMatrixFormat);
    tensorDesc.SetDataType(ge::DT_UINT8);
    tensorDesc.SetOriginDataType(ge::DT_UINT8);
    FUSION_PASS_MAKE_SHARED(
      (assitPtr = std::make_shared<ge::GeTensor>(tensorDesc, reinterpret_cast<uint8_t *>(inputAssit.get()),
                                                 dimNums * sizeof(uint8_t))),
      assitPtr = nullptr;
      return PARAM_INVALID);
  } else {
    return NOT_CHANGED;
  }
  // check op support
  vector<ge::GeTensorPtr> weights = {assitPtr};
  ge::OpDescUtils::SetWeights(triuVNode, weights);
  auto constInputNodes = OpDescUtils::GetConstInputs(triuVNode);
  NodePtr constInput = constInputNodes[0];
  constInput->GetOpDesc()->SetType(CONSTANTOP);
  triuDesc->SetType("Mul");

  return SUCCESS;
}

REGISTER_PASS("TriuFusionPass", BUILT_IN_GRAPH_PASS, TriuFusionPass);
}  // namespace fe
