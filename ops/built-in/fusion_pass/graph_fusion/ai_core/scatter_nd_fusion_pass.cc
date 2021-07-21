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
 * \file scatter_nd_fusion_pass.cpp
 * \brief
 */
#include "scatter_nd_fusion_pass.h"

#include <iostream>
#include <vector>
#include <map>

#include "graph/utils/op_desc_utils.h"
#include "graph/utils/graph_utils.h"
#include "graph/utils/node_utils.h"
#include "graph/utils/attr_utils.h"
#include "graph/debug/ge_attr_define.h"
#include "op_log.h"
#include "error_util.h"
#include "graph_optimizer/graph_fusion/fusion_pass_manager/fusion_pass_registry.h"
#include "securec.h"
#include "pattern_fusion_util.h"

using namespace std;
using namespace ge;

namespace fe {
static const string PATTERN_SCATTERND = "ScatterNd";
static const char* SCATTERND = "ScatterNd";
vector<FusionPattern*> ScatterNdFusionPass::DefinePatterns() {
  vector<FusionPattern*> patterns;
  FusionPattern* pattern = new (std::nothrow) FusionPattern("ScatterNdFusion");
  FUSION_PASS_CHECK(pattern == nullptr, VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "new a pattern object failed."),
                    return patterns);

  pattern->AddOpDesc(PATTERN_SCATTERND, {SCATTERND}).SetOutput(PATTERN_SCATTERND);

  patterns.push_back(pattern);

  return patterns;
}

Status ScatterNdFusionPass::Fusion(ge::ComputeGraph& graph, Mapping& mapping, vector<ge::NodePtr>& fusionNodes) {
  ge::NodePtr ScatterNdNode = GetNodeFromMapping(PATTERN_SCATTERND, mapping);
  FUSION_PASS_CHECK(ScatterNdNode == nullptr, VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "ScatterNdNode is null, fusion failed."),
                    return PARAM_INVALID);
  ge::OpDescPtr ScatterNdDesc = ScatterNdNode->GetOpDesc();
  FUSION_PASS_CHECK(ScatterNdDesc == nullptr,
                    VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "ScatterNdNode's OpDesc is null, fusion failed."),
                    return PARAM_INVALID);

  // check op support for dynamic scatter_nd
  FUSION_PASS_CHECK(CheckOpSupported(ScatterNdDesc), OP_LOGI(FUSED_OP_TYPE.c_str(), "Op ScatterNd Supported."),
                    return NOT_CHANGED);

  std::vector<PassAttrInfo> attrInfos = {{2, "shape", "SetListInt"}};
  const std::string fusionOpType = "ScatterNdD";
  ge::OpDescPtr fusionDescPtr = PatternFusionUtil::GetFusionOpDesc(ScatterNdNode, fusionOpType, attrInfos);
  FUSION_PASS_CHECK(fusionDescPtr == nullptr, OP_LOGI(FUSED_OP_TYPE.c_str(), "Fusion OP Desc is nullptr."),
                    return NOT_CHANGED);
  FUSION_PASS_CHECK(!CheckOpSupported(fusionDescPtr), OP_LOGI(FUSED_OP_TYPE.c_str(), "Op Not Supported."),
                    return NOT_CHANGED);

  ge::GeTensorDesc indicesTensor = ScatterNdNode->GetOpDesc()->GetInputDesc(0);
  ge::GeTensorDesc updateTensor = ScatterNdNode->GetOpDesc()->GetInputDesc(1);
  std::vector<int64_t> indicesShape = indicesTensor.GetShape().GetDims();
  std::vector<int64_t> updateShape = updateTensor.GetShape().GetDims();
  int64_t indicesDimNum = indicesTensor.GetShape().GetDimNum();
  int64_t updateDimNum = updateTensor.GetShape().GetDimNum();

  int64_t indicesLen = 1;
  for (int64_t i = 0; i < indicesDimNum - 1; i++) {
    if (PatternFusionUtil::IsUnknownShape(indicesShape[i])) {
      OP_LOGI(FUSED_OP_TYPE.c_str(), "ScatterNdFusionPass cannot be applied for unknown shape.");
      return NOT_CHANGED;
    }
    indicesLen = indicesLen * indicesShape[i];
  }
  int64_t updateSize = 1;
  for (int64_t i = 0; i < updateDimNum; i++) {
    if (PatternFusionUtil::IsUnknownShape(updateShape[i])) {
      OP_LOGI(FUSED_OP_TYPE.c_str(), "ScatterNdFusionPass cannot be applied for unknown shape.");
      return NOT_CHANGED;
    }
    updateSize = updateSize * updateShape[i];
  }
  if (indicesLen == 0) {
    OP_LOGI(FUSED_OP_TYPE.c_str(), "indicesShape contains invalid value 0.");
    return NOT_CHANGED;
  }
  int64_t updataSlice = updateSize / indicesLen;
  int64_t dataSize = 0;
  DataType dataType = updateTensor.GetDataType();
  if (dataType == ge::DT_INT32 || dataType == ge::DT_FLOAT) {
    dataSize = 4;
  } else if (dataType == ge::DT_FLOAT16) {
    dataSize = 2;
  } else if (dataType == ge::DT_INT8 || dataType == ge::DT_UINT8) {
    dataSize = 1;
  }

  if (dataSize > 0 && updataSlice > 0 && updataSlice < (32 / dataSize)) {
    OP_LOGI(FUSED_OP_TYPE.c_str(), "ScatterNdD update slice < 32byte, graph not changed.");
    return NOT_CHANGED;
  }

  Operator op = ge::OpDescUtils::CreateOperatorFromNode(ScatterNdNode);
  Tensor const_tensor1;
  if (GRAPH_SUCCESS != op.GetInputConstData("shape", const_tensor1)) {
    OP_LOGI(FUSED_OP_TYPE.c_str(), "ScatterNd GetInputConstData of shape failed.");
    return NOT_CHANGED;
  }

  ge::NodePtr fusion_node = nullptr;
  Status ret = PatternFusionUtil::ConstToAttrWithNode(graph, ScatterNdNode, fusionOpType, attrInfos, fusion_node);

  if (ret != SUCCESS) {
    OP_LOGI(FUSED_OP_TYPE.c_str(), "Scatternd has input which is not a CONST, graph not changed.");
    return NOT_CHANGED;
  }

  fusionNodes.push_back(fusion_node);
  return SUCCESS;
}

REGISTER_PASS("ScatterNdFusionPass", BUILT_IN_GRAPH_PASS, ScatterNdFusionPass);
}  // namespace fe
