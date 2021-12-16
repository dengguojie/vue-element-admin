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
 * \file fc_transdata_merge_fusion_pass.cpp
 * \brief
 */
#include "fc_transdata_merge_fusion_pass.h"

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
#include "pattern_fusion_util.h"
#include "graph_optimizer/graph_fusion/fusion_pass_manager/fusion_pass_registry.h"
#include "securec.h"

using namespace std;
using namespace ge;

namespace fe {
static const string PATTERN_TRANSDATA_1 = "Transdata_1";
static const string PATTERN_TRANSDATA_2 = "Transdata_2";
static const string PATTERN_RESHAPE = "Reshape";
static const string PATTERN_REFORMAT = "ReFormat";
static const string PATTERN_UNSQUEEZE_V2 = "UnsqueezeV2";
static const int NUM_2 = 2;
static const int NUM_4 = 4;
/*
 transdata(NZ) -->ReFormat->(Reshape)->transdata(NC1HWC0)
*/

vector<FusionPattern*> FCTransdataMergePass::DefinePatterns() {
  OP_LOGI(FUSED_OP_TYPE.c_str(), "Define FCTransdataMergePass pattern1 begin");
  vector<FusionPattern*> patterns;
  FusionPattern* pattern1 = new (std::nothrow) FusionPattern("FCTransdataMergePass1");
  FUSION_PASS_CHECK(pattern1 == nullptr, VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "new an object failed"),
                    return patterns);

  pattern1->AddOpDesc(PATTERN_TRANSDATA_1, {"TransData"})
      .AddOpDesc(PATTERN_REFORMAT, {"ReFormat"})
      .AddOpDesc(PATTERN_RESHAPE, {"Reshape"})
      .AddOpDesc(PATTERN_TRANSDATA_2, {"TransData"})
      .SetInputs(PATTERN_REFORMAT, {PATTERN_TRANSDATA_1})
      .SetInputs(PATTERN_RESHAPE, {PATTERN_REFORMAT})
      .SetInputs(PATTERN_TRANSDATA_2, {PATTERN_RESHAPE})
      .SetOutput(PATTERN_TRANSDATA_2);
  patterns.push_back(pattern1);
  OP_LOGI(FUSED_OP_TYPE.c_str(), "Define FCTransdataMergePass1 pattern end");

  FusionPattern* pattern2 = new (std::nothrow) FusionPattern("FCTransdataMergePass2");
  FUSION_PASS_CHECK(pattern2 == nullptr, VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "new an object failed"),
                    return patterns);
  pattern2->AddOpDesc(PATTERN_TRANSDATA_1, {"TransData"})
      .AddOpDesc(PATTERN_REFORMAT, {"ReFormat"})
      .AddOpDesc(PATTERN_UNSQUEEZE_V2, {"UnsqueezeV2"})
      .AddOpDesc(PATTERN_TRANSDATA_2, {"TransData"})
      .SetInputs(PATTERN_REFORMAT, {PATTERN_TRANSDATA_1})
      .SetInputs(PATTERN_UNSQUEEZE_V2, {PATTERN_REFORMAT})
      .SetInputs(PATTERN_TRANSDATA_2, {PATTERN_UNSQUEEZE_V2})
      .SetOutput(PATTERN_TRANSDATA_2);
  patterns.push_back(pattern2);
  OP_LOGI(FUSED_OP_TYPE.c_str(), "Define FCTransdataMergePass2 pattern end");

  return patterns;
}

Status FCTransdataMergePass::Fusion(ge::ComputeGraph& graph, Mapping& mapping, vector<ge::NodePtr>& fusionNodes) {
  OP_LOGI(FUSED_OP_TYPE.c_str(), "Define FCTransdataMergePass fusion begin");
  ge::NodePtr transData_1 = GetNodeFromMapping(PATTERN_TRANSDATA_1, mapping);
  ge::NodePtr reFormat = GetNodeFromMapping(PATTERN_REFORMAT, mapping);
  ge::NodePtr reShape = GetNodeFromMapping(PATTERN_UNSQUEEZE_V2, mapping);
  ge::NodePtr transData_2 = GetNodeFromMapping(PATTERN_TRANSDATA_2, mapping);

  if (reShape == nullptr) {
    reShape = GetNodeFromMapping(PATTERN_RESHAPE, mapping);
  }

  FUSION_PASS_CHECK(transData_1 == nullptr,
                    VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "transData_1 is null"),
                    return PARAM_INVALID);
  FUSION_PASS_CHECK(reFormat == nullptr, VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "reFormat is null"),
                    return PARAM_INVALID);
  FUSION_PASS_CHECK(reShape == nullptr, VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "reShape is null"),
                    return PARAM_INVALID);
  FUSION_PASS_CHECK(transData_2 == nullptr,
                    VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "transData_2 is null"),
                    return PARAM_INVALID);

  ge::OpDescPtr secondTransDataOpDesc = transData_2->GetOpDesc();
  FUSION_PASS_CHECK(secondTransDataOpDesc == nullptr, VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(),
                    "transData_2 opdesc is null"), return PARAM_INVALID);
  ge::GeTensorDesc secondTransDataOutputTensor = secondTransDataOpDesc->GetOutputDesc(0);
  if (secondTransDataOutputTensor.GetDataType() != ge::DT_FLOAT16 &&
      secondTransDataOutputTensor.GetDataType() != ge::DT_INT8) {
    OP_LOGI(FUSED_OP_TYPE.c_str(),
            "Second transdata output data type is not fp16 or int8 ,not support fusion, fusion end");
    return SUCCESS;
  }

  if (secondTransDataOutputTensor.GetFormat() != ge::FORMAT_NC1HWC0) {
    OP_LOGI(FUSED_OP_TYPE.c_str(),
            "Second transdata output format is not NC1HWC0 ,not support fusion, FCTransdataMergePass fusion end");
    return SUCCESS;
  }
  ge::GeTensorDesc secondTransDataInputTensor = secondTransDataOpDesc->GetInputDesc(0);
  if (secondTransDataInputTensor.GetFormat() != ge::FORMAT_NCHW &&
      secondTransDataInputTensor.GetFormat() != ge::FORMAT_ND) {
    OP_LOGI(FUSED_OP_TYPE.c_str(),
            "Second transdata input format is not NCHW or ND ,not support fusion, FCTransdataMergePass fusion end");
    return SUCCESS;
  }

  ge::OpDescPtr reshapeOpDesc = reShape->GetOpDesc();
  FUSION_PASS_CHECK(reshapeOpDesc == nullptr,
                    VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "Reshape opdesc is null"),
                    return PARAM_INVALID);
  ge::GeTensorDesc reshapeInputTensor = reshapeOpDesc->GetInputDesc(0);
  ge::GeTensorDesc reshapeOutputTensor = reshapeOpDesc->GetOutputDesc(0);
  ge::GeShape reshapeInputShape = reshapeInputTensor.GetShape();
  auto xShape = reshapeInputShape.GetDims();
  ge::GeShape reshapeOutputShape = reshapeOutputTensor.GetShape();
  auto yShape = reshapeOutputShape.GetDims();
  if ((xShape.size() != NUM_2) || (yShape.size() != NUM_4)) {
    OP_LOGI(FUSED_OP_TYPE.c_str(), "reshape input and output dims is not match, FCTransdataMergePass fusion end");
    return SUCCESS;
  }
  int64_t reshapeInputDim0Value = xShape[0];
  int64_t reshapeInputDim1Value = xShape[1];
  int64_t reshapeOutputDim0Value = yShape[0];
  int64_t reshapeOutputDim1Value = yShape[1];
  if (reshapeInputDim0Value != reshapeOutputDim0Value) {
    OP_LOGI(FUSED_OP_TYPE.c_str(), "reshapeInputDim0Value is not same with reshapeOutputDim0Value, fusion end");
    return SUCCESS;
  }
  if (reshapeInputDim1Value != reshapeOutputDim1Value) {
    OP_LOGI(FUSED_OP_TYPE.c_str(), "reshapeInputDim1Value is not same with reshapeOutputDim1Value, fusion end");
    return SUCCESS;
  }

  ge::OpDescPtr firstTransDataOpDesc = transData_1->GetOpDesc();
  FUSION_PASS_CHECK(firstTransDataOpDesc == nullptr, VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(),
                    "transData_1 opdesc is null"), return PARAM_INVALID);
  ge::GeTensorDesc firstTransDataInputTensor = firstTransDataOpDesc->GetInputDesc(0);
  if (static_cast<ge::Format>(ge::GetPrimaryFormat(firstTransDataInputTensor.GetFormat())) != ge::FORMAT_FRACTAL_NZ) {
    OP_LOGI(FUSED_OP_TYPE.c_str(),
            "firstTransDataInputTensor format is not FRACTAL_NZ, FCTransdataMergePass fusion end");
    return SUCCESS;
  }
  ge::GeTensorDesc firstTransDataOutputTensor = firstTransDataOpDesc->GetOutputDesc(0);
  if (firstTransDataOutputTensor.GetFormat() != ge::FORMAT_ND) {
    OP_LOGI(FUSED_OP_TYPE.c_str(), "firstTransDataOutputTensor format is not ND, FCTransdataMergePass fusion end");
    return SUCCESS;
  }
  FUSION_PASS_CHECK(transData_1->GetOpDesc()->UpdateOutputDesc(0, secondTransDataOutputTensor) != ge::GRAPH_SUCCESS,
                    VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "Update output desc fail."), return FAILED);
  ge::AttrUtils::SetStr(transData_1->GetOpDesc(), "dst_format", "NC1HWC0");
  // delete transData_2 node
  FUSION_PASS_CHECK(graph.RemoveNode(reFormat) != ge::GRAPH_SUCCESS,
                    VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "remove node failed"), return FAILED);
  FUSION_PASS_CHECK(graph.RemoveNode(reShape) != ge::GRAPH_SUCCESS,
                    VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "remove node failed"), return FAILED);
  FUSION_PASS_CHECK(graph.RemoveNode(transData_2) != ge::GRAPH_SUCCESS,
                    VECTOR_FUSION_INNER_ERR_REPORT(FUSED_OP_TYPE.c_str(), "remove node failed"), return FAILED);
  fusionNodes.push_back(transData_1);
  OP_LOGI(FUSED_OP_TYPE.c_str(), "FCTransdataMergePass fusion end");
  return SUCCESS;
}

REGISTER_PASS("FCTransdataMergePass", SECOND_ROUND_BUILT_IN_GRAPH_PASS, FCTransdataMergePass);
}  // namespace fe
