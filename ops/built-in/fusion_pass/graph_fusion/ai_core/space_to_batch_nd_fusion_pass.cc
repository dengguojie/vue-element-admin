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
 * \file space_to_batch_nd_fusion_pass.cpp
 * \brief
 */
#include "space_to_batch_nd_fusion_pass.h"
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
#include "graph_optimizer/graph_fusion/fusion_pass_manager/fusion_pass_registry.h"
#include "pattern_fusion_util.h"

using namespace ge;
namespace fe {
static const char* FUSED_NODE = "SpaceToBatchND";
static const std::string PATTERN_FUSEDNODE = "FusedNodeSpaceToBatchND";

static void CalcData(const Tensor& data, const DataType& dtype, std::vector<int64_t>& const_vec) {
  const uint8_t* constData = data.GetData();
  if (constData == nullptr) {
    return;
  }
  size_t size;
  if (dtype == ge::DT_INT32) {
    size = data.GetSize() / sizeof(int32_t);
    for (size_t i = 0; i < size; ++i) {
      const_vec.push_back(*((int32_t*)constData + i));
    }
  } else {
    size = data.GetSize() / sizeof(int64_t);
    for (size_t i = 0; i < size; ++i) {
      const_vec.push_back(*((int64_t*)constData + i));
    }
  }
}

vector<FusionPattern*> ConstToAttrSpaceToBatchNdPass::DefinePatterns() {
  vector<FusionPattern*> patterns;
  FusionPattern* pattern = new (std::nothrow) FusionPattern("ConstToAttrSpaceToBatchNdFusion");
  FUSION_PASS_CHECK(pattern == nullptr, OP_LOGE(FUSED_OP_TYPE.c_str(), "new a pattern object failed."),
                    return patterns);
  pattern->AddOpDesc(PATTERN_FUSEDNODE, {FUSED_NODE}).SetOutput(PATTERN_FUSEDNODE);
  patterns.push_back(pattern);
  return patterns;
}

Status ConstToAttrSpaceToBatchNdPass::Fusion(ge::ComputeGraph& graph, Mapping& mapping,
                                             vector<ge::NodePtr>& fusionNodes) {
  // get fused node
  ge::NodePtr fusedNode = GetNodeFromMapping(PATTERN_FUSEDNODE, mapping);
  FUSION_PASS_CHECK(fusedNode == nullptr, OP_LOGE(FUSED_OP_TYPE.c_str(), "fusedNode is null, fusion failed."),
                    return PARAM_INVALID);

  // build attr infos
  std::string fusionOpType = "SpaceToBatchNDD";
  std::vector<PassAttrInfo> attrInfos;
  PassAttrInfo block_shape = {1, "block_shape", "SetListInt"};
  attrInfos.push_back(block_shape);
  PassAttrInfo paddings = {2, "paddings", "SetListInt"};
  attrInfos.push_back(paddings);

  // build a fusion node op desc
  ge::OpDescPtr fusionDescPtr = PatternFusionUtil::GetFusionOpDesc(fusedNode, fusionOpType, attrInfos);
  FUSION_PASS_CHECK(fusionDescPtr == nullptr, OP_LOGE(FUSED_OP_TYPE.c_str(), "Fusion OP Desc is nullptr."),
                    return PARAM_INVALID);

  // check op support
  FUSION_PASS_CHECK(!CheckOpSupported(fusionDescPtr),
                    OP_LOGI(FUSED_OP_TYPE.c_str(), "Op SpaceToBatchNd Not Supported."), return NOT_CHANGED);

  ge::GeTensorDesc first_input_tensor = fusedNode->GetOpDesc()->GetInputDesc(0);
  if ((first_input_tensor.GetFormat() != ge::FORMAT_NHWC) && (first_input_tensor.GetFormat() != ge::FORMAT_NCHW)) {
    OP_LOGI(FUSED_OP_TYPE.c_str(),
            "SpaceToBatchND has input which format is not FORMAT_NHWC of FORMAT_NCHW, graph not changed.");
    return NOT_CHANGED;
  }
  size_t first_dim_num = first_input_tensor.GetShape().GetDimNum();
  if (first_dim_num != 4) {
    OP_LOGI(FUSED_OP_TYPE.c_str(), "SpaceToBatchND has first input which size is not 4, graph not changed.");
    return NOT_CHANGED;
  }

  // check const
  Operator op_fusion = ge::OpDescUtils::CreateOperatorFromNode(fusedNode);
  Tensor block_tensor;
  if (op_fusion.GetInputConstData("block_shape", block_tensor) != GRAPH_SUCCESS) {
    OP_LOGI(FUSED_OP_TYPE.c_str(),
            "SpaceToBatchND has input of block_shape which is not a constant, graph not changed.");
    return NOT_CHANGED;
  }
  DataType block_type = op_fusion.GetInputDesc("block_shape").GetDataType();
  std::vector<int64_t> block_vec;
  CalcData(block_tensor, block_type, block_vec);

  Tensor paddings_tensor;
  if (op_fusion.GetInputConstData("paddings", paddings_tensor) != GRAPH_SUCCESS) {
    OP_LOGI(FUSED_OP_TYPE.c_str(), "SpaceToBatchND has input of paddings which is not a constant, graph not changed.");
    return NOT_CHANGED;
  }
  DataType paddings_type = op_fusion.GetInputDesc("paddings").GetDataType();
  std::vector<int64_t> paddings_vec;
  CalcData(paddings_tensor, paddings_type, paddings_vec);

  if (first_input_tensor.GetFormat() == ge::FORMAT_NHWC) {
    if (block_vec.size() != 2) {
      OP_LOGI(FUSED_OP_TYPE.c_str(),
              "SpaceToBatchND has input of block_shape which size is not 2 by using NHWC, graph not changed.");
      return NOT_CHANGED;
    }
    if (paddings_vec.size() != 4) {
      OP_LOGI(FUSED_OP_TYPE.c_str(),
              "SpaceToBatchND has input of paddings which size is not 4 by using NHWC, graph not changed.");
      return NOT_CHANGED;
    }
  } else {
    if (block_vec.size() != 3) {
      OP_LOGI(FUSED_OP_TYPE.c_str(),
              "SpaceToBatchND has input of block_shape which size is not 3 by using NCHW, graph not changed.");
      return NOT_CHANGED;
    }
    if (paddings_vec.size() != 6) {
      OP_LOGI(FUSED_OP_TYPE.c_str(),
              "SpaceToBatchND has input of paddings which size is not 6 by using NCHW, graph not changed.");
      return NOT_CHANGED;
    }
    if (block_vec[0] != 1) {
      OP_LOGI(FUSED_OP_TYPE.c_str(),
              "SpaceToBatchND has input of block_shape which first value is not 1 by using NCHW, graph not changed.");
      return NOT_CHANGED;
    }
    if (paddings_vec[0] != 0 || paddings_vec[1] != 0) {
      OP_LOGI(FUSED_OP_TYPE.c_str(),
              "SpaceToBatchND has input of paddings which first value is not 0 by using NCHW, graph not changed.");
      return NOT_CHANGED;
    }
  }

  // const to attr
  ge::NodePtr fusionNode = nullptr;
  Status ret = PatternFusionUtil::ConstToAttrWithNode(graph, fusedNode, fusionOpType, attrInfos, fusionNode);
  if (ret != SUCCESS) {
    OP_LOGI(FUSED_OP_TYPE.c_str(), "SpaceToBatchND has input which is not a constant, graph not changed.");
    return NOT_CHANGED;
  }
  fusionNodes.push_back(fusionNode);
  OP_LOGI(FUSED_OP_TYPE.c_str(), "SpaceToBatchNDD fusion SUCCESSS!!!!!");
  return SUCCESS;
}
REGISTER_PASS("ConstToAttrSpaceToBatchNdFusion", BUILT_IN_GRAPH_PASS, ConstToAttrSpaceToBatchNdPass);
}  // namespace fe
