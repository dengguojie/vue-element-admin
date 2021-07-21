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
 * \file matrix_diag_part_fusion_pass.cpp
 * \brief
 */
#include "matrix_diag_part_fusion_pass.h"

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
#include "securec.h"

#include "op_log.h"
#include "error_util.h"
#include "fp16_t.hpp"
#include "pattern_fusion_util.h"

using namespace std;
using namespace ge;

namespace fe {
// the name of tf op
static const float FLOAT_NUM_ZERO = 0;
static const int32_t INT_NUM_ZERO = 0;
static const uint16_t UINT_NUM_ZERO = 0;
static const int8_t INT8_NUM_ZERO = 0;
static const uint8_t UINT8_NUM_ZERO = 0;
static const string PATTERN_MATRIXDIAGPART = "MatrixDiagPart";
static const std::string CONSTANTOP = "Constant";
static const char* MATRIXDIAGPART = "MatrixDiagPart";

template <typename Dtype>
Status AssitHelp(const int32_t n, const int32_t m, const int32_t x, Dtype& output1) {
  Dtype* output = &output1;
  int32_t z = m * x;
  if (z == INT_NUM_ZERO){
    return SUCCESS;
  }
  int32_t y = n / z;
  if (m > x) {
    for (int32_t i = 0; i < y; ++i) {
      for (int32_t j = 0; j < x; ++j) {
        output[(1 + m) * j + m * x * i] = 1;
      }
    }
  } else {
    for (int32_t i = 0; i < y; ++i) {
      for (int32_t j = 0; j < m; ++j) {
        output[(1 + m) * j + m * x * i] = 1;
      }
    }
  }

  return SUCCESS;
}

template <typename Dtype>
Status AssitHelp1(const int32_t n, const int32_t m, const int32_t x, Dtype& output1) {
  Dtype* output = &output1;
  fp16_t t;
  t.val = 0;
  int32_t xx = 1;
  t = xx;
  int32_t z = m * x;
  if (z == INT_NUM_ZERO){
    return SUCCESS;
  }
  int32_t y = n / z;
  if (m > x) {
    for (int32_t i = 0; i < y; ++i) {
      for (int32_t j = 0; j < x; ++j) {
        output[(1 + m) * j + m * x * i] = t.val;
      }
    }
  } else {
    for (int32_t i = 0; i < y; ++i) {
      for (int32_t j = 0; j < m; ++j) {
        output[(1 + m) * j + m * x * i] = t.val;
      }
    }
  }

  return SUCCESS;
}

vector<FusionPattern*> MatrixDiagPartFusionPass::DefinePatterns() {
  vector<FusionPattern*> patterns;

  // matrix_diag_part->matrix_diag_part_d
  // define MatrixDiagPartFusion
  FusionPattern* pattern = new (std::nothrow) FusionPattern("MatrixDiagPartFusionPass");
  FUSION_PASS_CHECK(pattern == nullptr, VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "new a pattern object failed."),
                    return patterns);

  // define origin graph
  pattern->AddOpDesc(PATTERN_MATRIXDIAGPART, {MATRIXDIAGPART}).SetOutput(PATTERN_MATRIXDIAGPART);

  patterns.push_back(pattern);

  return patterns;
}

Status MatrixDiagPartFusionPass::Fusion(ge::ComputeGraph& graph, Mapping& mapping, vector<ge::NodePtr>& fusionNodes) {
  // matrix_diag_part node
  ge::NodePtr matrixdiagpartVNode = GetNodeFromMapping(PATTERN_MATRIXDIAGPART, mapping);
  FUSION_PASS_CHECK(matrixdiagpartVNode == nullptr,
                    VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(),
                            "matrixdiagpartVNode is null, "
                            "fusion failed."),
                    return PARAM_INVALID);

  // input of matrix_diag_part
  ge::OpDescPtr matrixdiagpartDesc = matrixdiagpartVNode->GetOpDesc();
  FUSION_PASS_CHECK(matrixdiagpartDesc == nullptr,
                    VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(),
                            "matrixdiagpartVNode's OpDesc is "
                            "null, fusion failed."),
                    return PARAM_INVALID);
  ge::OpDescPtr fusionDesc = AttrUtils::CopyOpDesc(matrixdiagpartDesc);

  // get the input desc of the entrance of matrix_diag_part node to differentiate between const and var
  ge::GeTensorDesc matrixdiagpartInputTensor = matrixdiagpartVNode->GetOpDesc()->GetInputDesc(0);

  // get the shape info
  ge::GeShape matrixdiagpartInputShape = matrixdiagpartInputTensor.GetShape();

  // get the data type
  DataType dataType = matrixdiagpartInputTensor.GetDataType();

  // multiples of dims
  int64_t dimNums = 1;
  int64_t dimsInput = matrixdiagpartInputShape.GetDimNum() - 1;
  int64_t dimsInput1 = matrixdiagpartInputShape.GetDimNum() - 2;
  for (size_t j = 0; j < matrixdiagpartInputShape.GetDimNum(); ++j) {
    if (PatternFusionUtil::IsUnknownShape(matrixdiagpartInputShape.GetDim(j))) {
      VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "MatrixDiagPartFusionPass cannot be applied for unknown shape.");
      return NOT_CHANGED;
    }
    dimNums = matrixdiagpartInputShape.GetDim(j) * dimNums;
  }

  // get the last dims of input shape
  int64_t dimNums1 = matrixdiagpartInputShape.GetDim(dimsInput);
  int64_t dimNums2 = matrixdiagpartInputShape.GetDim(dimsInput1);

  if (PatternFusionUtil::IsUnknownShape(dimNums1) ||
      PatternFusionUtil::IsUnknownShape(dimNums2)) {
    VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "MatrixDiagPartFusionPass cannot be applied for unknown shape.");
    return NOT_CHANGED;
  }

  vector<int64_t> dimInfo = matrixdiagpartInputShape.GetDims();

  Format assitMatrixFormat = matrixdiagpartInputTensor.GetFormat();

  ge::GeTensorPtr assitPtr = nullptr;
  ge::GeTensorDesc tensorDesc(GeShape(), ge::FORMAT_ND, ge::DT_FLOAT);
  if (dataType == ge::DT_FLOAT) {
    unique_ptr<float[]> inputAssit(new (std::nothrow) float[dimNums]());
    FUSION_PASS_CHECK(inputAssit.get() == nullptr, VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "inputAssit is NULL"),
                      return PARAM_INVALID);
    Status ret = NnSet(dimNums, FLOAT_NUM_ZERO, *reinterpret_cast<float*>(inputAssit.get()));
    FUSION_PASS_CHECK(ret != SUCCESS, OP_LOGW(FUSED_OP_TYPE.c_str(), "NnSet failed."), return NOT_CHANGED);

    ret = AssitHelp(dimNums, dimNums1, dimNums2, *inputAssit.get());
    FUSION_PASS_CHECK(ret != SUCCESS, OP_LOGW(FUSED_OP_TYPE.c_str(), "AssitHelp failed."), return NOT_CHANGED);

    // define the shape of auxiliary matrix
    ge::GeShape assitShape = matrixdiagpartInputShape;
    tensorDesc.SetShape(assitShape);
    tensorDesc.SetOriginShape(assitShape);
    tensorDesc.SetFormat(assitMatrixFormat);
    tensorDesc.SetOriginFormat(assitMatrixFormat);
    tensorDesc.SetDataType(ge::DT_FLOAT);
    tensorDesc.SetOriginDataType(ge::DT_FLOAT);
    FUSION_PASS_MAKE_SHARED((assitPtr = std::make_shared<ge::GeTensor>(
                                 tensorDesc, reinterpret_cast<uint8_t*>(inputAssit.get()), dimNums * sizeof(float))),
                            assitPtr = nullptr;
                            return PARAM_INVALID);
  } else if (dataType == ge::DT_INT32) {
    unique_ptr<int32_t[]> inputAssit(new (std::nothrow) int32_t[dimNums]());
    FUSION_PASS_CHECK(inputAssit.get() == nullptr, VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "inputAssit is NULL"),
                      return PARAM_INVALID);

    Status ret = NnSet(dimNums, INT_NUM_ZERO, *reinterpret_cast<int32_t*>(inputAssit.get()));
    FUSION_PASS_CHECK(ret != SUCCESS, OP_LOGW(FUSED_OP_TYPE.c_str(), "NnSet failed."), return NOT_CHANGED);

    ret = AssitHelp(dimNums, dimNums1, dimNums2, *inputAssit.get());
    FUSION_PASS_CHECK(ret != SUCCESS, OP_LOGW(FUSED_OP_TYPE.c_str(), "AssitHelp failed."), return NOT_CHANGED);

    // define the shape of auxiliary matrix
    ge::GeShape assitShape = matrixdiagpartInputShape;
    tensorDesc.SetShape(assitShape);
    tensorDesc.SetOriginShape(assitShape);
    tensorDesc.SetFormat(assitMatrixFormat);
    tensorDesc.SetOriginFormat(assitMatrixFormat);
    tensorDesc.SetDataType(ge::DT_INT32);
    tensorDesc.SetOriginDataType(ge::DT_INT32);
    FUSION_PASS_MAKE_SHARED((assitPtr = std::make_shared<ge::GeTensor>(
                                 tensorDesc, reinterpret_cast<uint8_t*>(inputAssit.get()), dimNums * sizeof(int32_t))),
                            assitPtr = nullptr;
                            return PARAM_INVALID);
  } else if (dataType == ge::DT_FLOAT16) {
    unique_ptr<uint16_t[]> inputAssit(new (std::nothrow) uint16_t[dimNums]());
    FUSION_PASS_CHECK(inputAssit.get() == nullptr, VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "inputAssit is NULL"),
                      return PARAM_INVALID);
    Status ret = NnSet(dimNums, UINT_NUM_ZERO, *reinterpret_cast<uint16_t*>(inputAssit.get()));
    FUSION_PASS_CHECK(ret != SUCCESS, OP_LOGW(FUSED_OP_TYPE.c_str(), "NnSet failed."), return NOT_CHANGED);

    ret = AssitHelp1(dimNums, dimNums1, dimNums2, *inputAssit.get());
    FUSION_PASS_CHECK(ret != SUCCESS, OP_LOGW(FUSED_OP_TYPE.c_str(), "AssitHelp failed."), return NOT_CHANGED);

    // define the shape of auxiliary matrix
    ge::GeShape assitShape = matrixdiagpartInputShape;
    tensorDesc.SetShape(assitShape);
    tensorDesc.SetOriginShape(assitShape);
    tensorDesc.SetFormat(assitMatrixFormat);
    tensorDesc.SetOriginFormat(assitMatrixFormat);
    tensorDesc.SetDataType(ge::DT_FLOAT16);
    tensorDesc.SetOriginDataType(ge::DT_FLOAT16);
    FUSION_PASS_MAKE_SHARED((assitPtr = std::make_shared<ge::GeTensor>(
                                 tensorDesc, reinterpret_cast<uint8_t*>(inputAssit.get()), dimNums * sizeof(uint16_t))),
                            assitPtr = nullptr;
                            return PARAM_INVALID);
  } else if (dataType == ge::DT_INT8) {
    unique_ptr<int8_t[]> inputAssit(new (std::nothrow) int8_t[dimNums]());
    FUSION_PASS_CHECK(inputAssit.get() == nullptr, VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "inputAssit is NULL"),
                      return PARAM_INVALID);

    Status ret = NnSet(dimNums, INT8_NUM_ZERO, *reinterpret_cast<int8_t*>(inputAssit.get()));
    FUSION_PASS_CHECK(ret != SUCCESS, OP_LOGW(FUSED_OP_TYPE.c_str(), "NnSet failed."), return NOT_CHANGED);

    ret = AssitHelp(dimNums, dimNums1, dimNums2, *inputAssit.get());
    FUSION_PASS_CHECK(ret != SUCCESS, OP_LOGW(FUSED_OP_TYPE.c_str(), "AssitHelp failed."), return NOT_CHANGED);

    // define the shape of auxiliary matrix
    ge::GeShape assitShape = matrixdiagpartInputShape;
    tensorDesc.SetShape(assitShape);
    tensorDesc.SetOriginShape(assitShape);
    tensorDesc.SetFormat(assitMatrixFormat);
    tensorDesc.SetOriginFormat(assitMatrixFormat);
    tensorDesc.SetDataType(ge::DT_INT8);
    tensorDesc.SetOriginDataType(ge::DT_INT8);
    FUSION_PASS_MAKE_SHARED((assitPtr = std::make_shared<ge::GeTensor>(
                                 tensorDesc, reinterpret_cast<uint8_t*>(inputAssit.get()), dimNums * sizeof(int8_t))),
                            assitPtr = nullptr;
                            return PARAM_INVALID);
  } else if (dataType == ge::DT_UINT8) {
    unique_ptr<uint8_t[]> inputAssit(new (std::nothrow) uint8_t[dimNums]());
    FUSION_PASS_CHECK(inputAssit.get() == nullptr, VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "inputAssit is NULL"),
                      return PARAM_INVALID);

    Status ret = NnSet(dimNums, UINT8_NUM_ZERO, *reinterpret_cast<uint8_t*>(inputAssit.get()));
    FUSION_PASS_CHECK(ret != SUCCESS, OP_LOGW(FUSED_OP_TYPE.c_str(), "NnSet failed."), return NOT_CHANGED);

    ret = AssitHelp(dimNums, dimNums1, dimNums2, *inputAssit.get());
    FUSION_PASS_CHECK(ret != SUCCESS, OP_LOGW(FUSED_OP_TYPE.c_str(), "AssitHelp failed."), return NOT_CHANGED);

    // define the shape of auxiliary matrix
    ge::GeShape assitShape = matrixdiagpartInputShape;
    tensorDesc.SetShape(assitShape);
    tensorDesc.SetOriginShape(assitShape);
    tensorDesc.SetFormat(assitMatrixFormat);
    tensorDesc.SetOriginFormat(assitMatrixFormat);
    tensorDesc.SetDataType(ge::DT_UINT8);
    tensorDesc.SetOriginDataType(ge::DT_UINT8);
    FUSION_PASS_MAKE_SHARED((assitPtr = std::make_shared<ge::GeTensor>(
                                 tensorDesc, reinterpret_cast<uint8_t*>(inputAssit.get()), dimNums * sizeof(uint8_t))),
                            assitPtr = nullptr;
                            return PARAM_INVALID);
  } else {
    return NOT_CHANGED;
  }
  // check op support
  vector<ge::GeTensorPtr> weights = {assitPtr};
  ge::OpDescUtils::SetWeights(matrixdiagpartVNode, weights);
  auto constInputNodes = OpDescUtils::GetConstInputs(matrixdiagpartVNode);
  FUSION_PASS_CHECK(constInputNodes.empty(), VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "const input empty."), return PARAM_INVALID);
  NodePtr constInput = constInputNodes[0];
  constInput->GetOpDesc()->SetType(CONSTANTOP);
  matrixdiagpartDesc->SetType("MatrixDiagPartD");

  return SUCCESS;
}

REGISTER_PASS("MatrixDiagPartFusionPass", BUILT_IN_GRAPH_PASS, MatrixDiagPartFusionPass);
}  // namespace fe
