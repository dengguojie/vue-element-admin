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
 * \file matrix_diag_fusion_pass.cpp
 * \brief
 */
#include "matrix_diag_fusion_pass.h"

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
static const int8_t INT8_NUM_ZERO = 0;
static const uint8_t UINT8_NUM_ZERO = 0;
static const string PATTERN_MATRIXDIAG = "MatrixDiag";
static const std::string CONSTANTOP = "Constant";
static const char* MATRIXDIAG = "MatrixDiag";

template <typename Dtype>
Status AssitHelp(const int32_t n, const int32_t m, Dtype& output1) {
  Dtype* output = &output1;
  for (int i = 0; i < m * n; ++i) {
    output[i * m + i % m] = 1;
  }
  return SUCCESS;
}

template <typename Dtype>
Status AssitHelp1(const int32_t n, const int32_t m, Dtype& output1) {
  Dtype* output = &output1;
  fp16_t t;
  t.val = 0;
  int32_t xx = 1;
  t = xx;
  for (int i = 0; i < m * n; ++i) {
    output[i * m + i % m] = t.val;
  }
  return SUCCESS;
}

vector<FusionPattern*> MatrixDiagFusionPass::DefinePatterns() {
  vector<FusionPattern*> patterns;

  // matrix_diag->matrix_diag_d
  // define MatrixDiagFusion
  FusionPattern* pattern = new (std::nothrow) FusionPattern("MatrixDiagFusionPass");
  FUSION_PASS_CHECK(pattern == nullptr, OP_LOGE(FUSED_OP_TYPE.c_str(), "new a pattern object failed."),
                    return patterns);

  // define origin graph
  pattern->AddOpDesc(PATTERN_MATRIXDIAG, {MATRIXDIAG}).SetOutput(PATTERN_MATRIXDIAG);

  patterns.push_back(pattern);

  return patterns;
}

Status MatrixDiagFusionPass::Fusion(ge::ComputeGraph& graph, Mapping& mapping, vector<ge::NodePtr>& fusionNodes) {
  // matrix_diag node
  ge::NodePtr matrixdiagVNode = GetNodeFromMapping(PATTERN_MATRIXDIAG, mapping);
  FUSION_PASS_CHECK(matrixdiagVNode == nullptr,
                    OP_LOGE(FUSED_OP_TYPE.c_str(),
                            "matrixdiagVNode is null, "
                            "fusion failed."),
                    return PARAM_INVALID);

  // input of matrix_diag
  ge::OpDescPtr matrixdiagDesc = matrixdiagVNode->GetOpDesc();
  FUSION_PASS_CHECK(matrixdiagDesc == nullptr,
                    OP_LOGE(FUSED_OP_TYPE.c_str(),
                            "matrixdiagVNode's OpDesc is "
                            "null, fusion failed."),
                    return PARAM_INVALID);
  ge::OpDescPtr fusionDesc = AttrUtils::CopyOpDesc(matrixdiagDesc);
  // find the parent node of matrix_diag
  ge::InDataAnchorPtr matrixdiagAnchorPtr0 = matrixdiagVNode->GetInDataAnchor(0);
  ge::OutDataAnchorPtr constAnchorPtr0 = matrixdiagAnchorPtr0->GetPeerOutAnchor();
  ge::NodePtr constNode0 = constAnchorPtr0->GetOwnerNode();

  // get the input desc of entrance node to differentiate const and varj
  ge::GeTensorDesc matrixdiagInputTensor = constNode0->GetOpDesc()->GetOutputDesc(0);

  // get the shape info
  ge::GeShape matrixdiagInputShape = matrixdiagInputTensor.GetShape();

  // get the data type
  DataType dataType = matrixdiagInputTensor.GetDataType();

  // multipies of dims
  int64_t dimNums = 1;
  int64_t dimsInput = matrixdiagInputShape.GetDimNum() - 1;
  for (size_t j = 0; j < matrixdiagInputShape.GetDimNum() - 1; ++j) {
    dimNums = matrixdiagInputShape.GetDim(j) * dimNums;
  }

  // get the last dim of input shape
  int64_t dimNums1 = matrixdiagInputShape.GetDim(dimsInput);
  vector<int64_t> dimInfo = matrixdiagInputShape.GetDims();

  Format assitMatrixFormat = matrixdiagInputTensor.GetFormat();

  std::vector<int64_t> dim_vec;
  for (size_t j = 0; j < matrixdiagInputShape.GetDimNum(); ++j) {
    dim_vec.push_back(dimInfo[j]);
  }
  dim_vec.push_back(dimNums1);
  ge::GeTensorDesc td;
  td.SetShape(ge::GeShape(dim_vec));
  ge::GeShape tdShape = td.GetShape();

  ge::GeTensorPtr assitPtr = nullptr;
  ge::GeTensorDesc tensorDesc(GeShape(), ge::FORMAT_ND, ge::DT_FLOAT);
  if (dataType == ge::DT_FLOAT) {
    unique_ptr<float[]> inputAssit(new (std::nothrow) float[dimNums * dimNums1 * dimNums1]());
    FUSION_PASS_CHECK(inputAssit.get() == nullptr, OP_LOGE(FUSED_OP_TYPE.c_str(), "inputAssit is NULL"),
                      return PARAM_INVALID);
    Status ret = NnSet(dimNums * dimNums1 * dimNums1, FLOAT_NUM_ZERO, *reinterpret_cast<float*>(inputAssit.get()));
    FUSION_PASS_CHECK(ret != SUCCESS, OP_LOGE(FUSED_OP_TYPE.c_str(), "NnSet failed."), return ret);

    ret = AssitHelp(dimNums, dimNums1, *inputAssit.get());
    FUSION_PASS_CHECK(ret != SUCCESS, OP_LOGE(FUSED_OP_TYPE.c_str(), "AssitHelp failed."), return ret);

    // define the shape of auxiliary matrix
    ge::GeShape assitShape = tdShape;
    tensorDesc.SetShape(assitShape);
    tensorDesc.SetOriginShape(assitShape);
    tensorDesc.SetFormat(assitMatrixFormat);
    tensorDesc.SetOriginFormat(assitMatrixFormat);
    tensorDesc.SetDataType(ge::DT_FLOAT);
    tensorDesc.SetOriginDataType(ge::DT_FLOAT);
    FUSION_PASS_MAKE_SHARED(
        (assitPtr = std::make_shared<ge::GeTensor>(tensorDesc, reinterpret_cast<uint8_t*>(inputAssit.get()),
                                                   dimNums * dimNums1 * dimNums1 * sizeof(float))),
        assitPtr = nullptr;
        return PARAM_INVALID);
  } else if (dataType == ge::DT_INT32) {
    unique_ptr<int32_t[]> inputAssit(new (std::nothrow) int32_t[dimNums * dimNums1 * dimNums1]());
    FUSION_PASS_CHECK(inputAssit.get() == nullptr, OP_LOGE(FUSED_OP_TYPE.c_str(), "inputAssit is NULL"),
                      return PARAM_INVALID);

    Status ret = NnSet(dimNums * dimNums1 * dimNums1, INT_NUM_ZERO, *reinterpret_cast<int32_t*>(inputAssit.get()));
    FUSION_PASS_CHECK(ret != SUCCESS, OP_LOGE(FUSED_OP_TYPE.c_str(), "NnSet failed."), return ret);

    ret = AssitHelp(dimNums, dimNums1, *inputAssit.get());
    FUSION_PASS_CHECK(ret != SUCCESS, OP_LOGE(FUSED_OP_TYPE.c_str(), "AssitHelp failed."), return ret);

    // define the shape of auxiliary matrix
    ge::GeShape assitShape = tdShape;
    tensorDesc.SetShape(assitShape);
    tensorDesc.SetOriginShape(assitShape);
    tensorDesc.SetFormat(assitMatrixFormat);
    tensorDesc.SetOriginFormat(assitMatrixFormat);
    tensorDesc.SetDataType(ge::DT_INT32);
    tensorDesc.SetOriginDataType(ge::DT_INT32);
    FUSION_PASS_MAKE_SHARED(
        (assitPtr = std::make_shared<ge::GeTensor>(tensorDesc, reinterpret_cast<uint8_t*>(inputAssit.get()),
                                                   dimNums * dimNums1 * dimNums1 * sizeof(int32_t))),
        assitPtr = nullptr;
        return PARAM_INVALID);
  } else if (dataType == ge::DT_FLOAT16) {
    unique_ptr<uint16_t[]> inputAssit(new (std::nothrow) uint16_t[dimNums * dimNums1 * dimNums1]());
    FUSION_PASS_CHECK(inputAssit.get() == nullptr, OP_LOGE(FUSED_OP_TYPE.c_str(), "inputAssit is NULL"),
                      return PARAM_INVALID);
    Status ret = NnSet(dimNums * dimNums1 * dimNums1, UINT_NUM_ZERO, *reinterpret_cast<uint16_t*>(inputAssit.get()));
    FUSION_PASS_CHECK(ret != SUCCESS, OP_LOGE(FUSED_OP_TYPE.c_str(), "NnSet failed."), return ret);

    ret = AssitHelp1(dimNums, dimNums1, *inputAssit.get());
    FUSION_PASS_CHECK(ret != SUCCESS, OP_LOGE(FUSED_OP_TYPE.c_str(), "AssitHelp failed."), return ret);

    // define the shape of auxiliary matrix
    ge::GeShape assitShape = tdShape;
    tensorDesc.SetShape(assitShape);
    tensorDesc.SetOriginShape(assitShape);
    tensorDesc.SetFormat(assitMatrixFormat);
    tensorDesc.SetOriginFormat(assitMatrixFormat);
    tensorDesc.SetDataType(ge::DT_FLOAT16);
    tensorDesc.SetOriginDataType(ge::DT_FLOAT16);
    FUSION_PASS_MAKE_SHARED(
        (assitPtr = std::make_shared<ge::GeTensor>(tensorDesc, reinterpret_cast<uint8_t*>(inputAssit.get()),
                                                   dimNums * dimNums1 * dimNums1 * sizeof(uint16_t))),
        assitPtr = nullptr;
        return PARAM_INVALID);
  } else if (dataType == ge::DT_INT8) {
    unique_ptr<int8_t[]> inputAssit(new (std::nothrow) int8_t[dimNums * dimNums1 * dimNums1]());
    FUSION_PASS_CHECK(inputAssit.get() == nullptr, OP_LOGE(FUSED_OP_TYPE.c_str(), "inputAssit is NULL"),
                      return PARAM_INVALID);

    Status ret = NnSet(dimNums * dimNums1 * dimNums1, INT8_NUM_ZERO, *reinterpret_cast<int8_t*>(inputAssit.get()));
    FUSION_PASS_CHECK(ret != SUCCESS, OP_LOGE(FUSED_OP_TYPE.c_str(), "NnSet failed."), return ret);

    ret = AssitHelp(dimNums, dimNums1, *inputAssit.get());
    FUSION_PASS_CHECK(ret != SUCCESS, OP_LOGE(FUSED_OP_TYPE.c_str(), "AssitHelp failed."), return ret);

    // define the shape of auxiliary matrix
    ge::GeShape assitShape = tdShape;
    tensorDesc.SetShape(assitShape);
    tensorDesc.SetOriginShape(assitShape);
    tensorDesc.SetFormat(assitMatrixFormat);
    tensorDesc.SetOriginFormat(assitMatrixFormat);
    tensorDesc.SetDataType(ge::DT_INT8);
    tensorDesc.SetOriginDataType(ge::DT_INT8);
    FUSION_PASS_MAKE_SHARED(
        (assitPtr = std::make_shared<ge::GeTensor>(tensorDesc, reinterpret_cast<uint8_t*>(inputAssit.get()),
                                                   dimNums * dimNums1 * dimNums1 * sizeof(int8_t))),
        assitPtr = nullptr;
        return PARAM_INVALID);
  } else if (dataType == ge::DT_UINT8) {
    unique_ptr<uint8_t[]> inputAssit(new (std::nothrow) uint8_t[dimNums * dimNums1 * dimNums1]());
    FUSION_PASS_CHECK(inputAssit.get() == nullptr, OP_LOGE(FUSED_OP_TYPE.c_str(), "inputAssit is NULL"),
                      return PARAM_INVALID);

    Status ret = NnSet(dimNums * dimNums1 * dimNums1, UINT8_NUM_ZERO, *reinterpret_cast<uint8_t*>(inputAssit.get()));
    FUSION_PASS_CHECK(ret != SUCCESS, OP_LOGE(FUSED_OP_TYPE.c_str(), "NnSet failed."), return ret);

    ret = AssitHelp(dimNums, dimNums1, *inputAssit.get());
    FUSION_PASS_CHECK(ret != SUCCESS, OP_LOGE(FUSED_OP_TYPE.c_str(), "AssitHelp failed."), return ret);

    // define the shape of auxiliary matrix
    ge::GeShape assitShape = tdShape;
    tensorDesc.SetShape(assitShape);
    tensorDesc.SetOriginShape(assitShape);
    tensorDesc.SetFormat(assitMatrixFormat);
    tensorDesc.SetOriginFormat(assitMatrixFormat);
    tensorDesc.SetDataType(ge::DT_UINT8);
    tensorDesc.SetOriginDataType(ge::DT_UINT8);
    FUSION_PASS_MAKE_SHARED(
        (assitPtr = std::make_shared<ge::GeTensor>(tensorDesc, reinterpret_cast<uint8_t*>(inputAssit.get()),
                                                   dimNums * dimNums1 * dimNums1 * sizeof(uint8_t))),
        assitPtr = nullptr;
        return PARAM_INVALID);
  }
  // check op support
  vector<ge::GeTensorPtr> weights = {assitPtr};
  ge::OpDescUtils::SetWeights(matrixdiagVNode, weights);
  auto constInputNodes = OpDescUtils::GetConstInputs(matrixdiagVNode);
  NodePtr constInput = constInputNodes[0];
  constInput->GetOpDesc()->SetType(CONSTANTOP);
  matrixdiagDesc->SetType("MatrixDiagD");
  return SUCCESS;
}

REGISTER_PASS("MatrixDiagFusionPass", BUILT_IN_GRAPH_PASS, MatrixDiagFusionPass);
}  // namespace fe
