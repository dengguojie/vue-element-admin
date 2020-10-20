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
 * \file batch_to_space_nd_fusion_pass.cpp
 * \brief BatchToSpaceND fusion pass(BatchToSpaceND --> BatchToSpaceNDD)
 */
#include "batch_to_space_nd_fusion_pass.h"
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
static const char* FUSED_NODE = "BatchToSpaceND";
static const std::string PATTERN_FUSEDNODE = "FusedNodeBatchToSpaceND";

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

vector<FusionPattern*> ConstToAttrBatchToSpaceNdPass::DefinePatterns() {
  vector<FusionPattern*> patterns;
  FusionPattern* pattern = new (std::nothrow) FusionPattern("ConstToAttrBatchToSpaceNdFusion");
  FUSION_PASS_CHECK(pattern == nullptr, OP_LOGE(FUSED_OP_TYPE.c_str(), "new a pattern object failed."),
                    return patterns);
  pattern->AddOpDesc(PATTERN_FUSEDNODE, {FUSED_NODE}).SetOutput(PATTERN_FUSEDNODE);
  patterns.push_back(pattern);
  return patterns;
}

// vector<ge::NodePtr> &fusionNodes: Store fusion nodes,
//       including newly added nodes and fused but not deleted nodes
Status ConstToAttrBatchToSpaceNdPass::Fusion(ge::ComputeGraph& graph, Mapping& mapping,
                                             vector<ge::NodePtr>& fusionNodes) {
  // get fused node
  ge::NodePtr fusedNode = GetNodeFromMapping(PATTERN_FUSEDNODE, mapping);
  FUSION_PASS_CHECK(fusedNode == nullptr, OP_LOGE(FUSED_OP_TYPE.c_str(), "fusedNode is null, fusion failed."),
                    return PARAM_INVALID);

  // build attr infos
  std::string fusionOpType = "BatchToSpaceNDD";
  std::vector<PassAttrInfo> attrInfos;
  PassAttrInfo block_shape = {1, "block_shape", "SetListInt"};
  attrInfos.push_back(block_shape);
  PassAttrInfo crops = {2, "crops", "SetListInt"};
  attrInfos.push_back(crops);

  // build a fusion node op desc
  ge::OpDescPtr fusionDescPtr = PatternFusionUtil::GetFusionOpDesc(fusedNode, fusionOpType, attrInfos);
  FUSION_PASS_CHECK(fusionDescPtr == nullptr, OP_LOGE(FUSED_OP_TYPE.c_str(), "Fusion OP Desc is nullptr."),
                    return PARAM_INVALID);

  // check op support
  FUSION_PASS_CHECK(!CheckOpSupported(fusionDescPtr),
                    OP_LOGI(FUSED_OP_TYPE.c_str(), "Op BatchToSpaceND Not Supported."), return NOT_CHANGED);

  ge::GeTensorDesc first_input_tensor = fusedNode->GetOpDesc()->GetInputDesc(0);
  if ((first_input_tensor.GetFormat() != ge::FORMAT_NHWC) && (first_input_tensor.GetFormat() != ge::FORMAT_NCHW)) {
    OP_LOGI(FUSED_OP_TYPE.c_str(),
            "BatchToSpaceND has input which format is not FORMAT_NHWC or FORMAT_NCHW, graph not changed.");
    return NOT_CHANGED;
  }
  size_t first_dim_num = first_input_tensor.GetShape().GetDimNum();
  if (first_dim_num != 4) {
    OP_LOGI(FUSED_OP_TYPE.c_str(), "BatchToSpaceND has first input which size is not 4, graph not changed.");
    return NOT_CHANGED;
  }

  // check const
  Operator op_fusion = ge::OpDescUtils::CreateOperatorFromNode(fusedNode);
  Tensor block_tensor;
  if (op_fusion.GetInputConstData("block_shape", block_tensor) != GRAPH_SUCCESS) {
    OP_LOGI(FUSED_OP_TYPE.c_str(),
            "BatchToSpaceND has input of block_shape which is not a constant, graph not changed.");
    return NOT_CHANGED;
  }
  DataType block_type = op_fusion.GetInputDesc("block_shape").GetDataType();
  std::vector<int64_t> block_vec;
  CalcData(block_tensor, block_type, block_vec);

  Tensor crops_tensor;
  if (op_fusion.GetInputConstData("crops", crops_tensor) != GRAPH_SUCCESS) {
    OP_LOGI(FUSED_OP_TYPE.c_str(), "BatchToSpaceND has input of crops which is not a constant, graph not changed.");
    return NOT_CHANGED;
  }
  DataType crops_type = op_fusion.GetInputDesc("crops").GetDataType();
  int64_t block_shape1 = 1;
  uint32_t block_shape2 = 2;
  uint32_t block_shape3 = 3;
  uint32_t crops_size4 = 4;
  uint32_t crops_size6 = 6;

  std::vector<int64_t> crops_vec;
  CalcData(crops_tensor, crops_type, crops_vec);

  if (first_input_tensor.GetFormat() == ge::FORMAT_NHWC) {
    if (block_vec.size() != block_shape2) {
      OP_LOGI(FUSED_OP_TYPE.c_str(),
              "BatchToSpaceND has input of block_shape which size is not 2 by using NHWC, graph not changed.");
      return NOT_CHANGED;
    }
    if (crops_vec.size() != crops_size4) {
      OP_LOGI(FUSED_OP_TYPE.c_str(),
              "BatchToSpaceND has input of crops which size is not 4 by using NHWC, graph not changed.");
      return NOT_CHANGED;
    }
  } else {
    if (block_vec.size() != block_shape3) {
      OP_LOGI(FUSED_OP_TYPE.c_str(),
              "BatchToSpaceND has input of block_shape which size is not 3 by using NCHW, graph not changed.");
      return NOT_CHANGED;
    }
    if (crops_vec.size() != crops_size6) {
      OP_LOGI(FUSED_OP_TYPE.c_str(),
              "BatchToSpaceND has input of crops which size is not 6 by using NCHW, graph not changed.");
      return NOT_CHANGED;
    }
    if (block_vec[0] != block_shape1) {
      OP_LOGI(FUSED_OP_TYPE.c_str(),
              "BatchToSpaceND has input of block_shape which first value is not 1 by using NCHW, graph not changed.");
      return NOT_CHANGED;
    }
    if (crops_vec[0] != 0 || crops_vec[1] != 0) {
      OP_LOGI(FUSED_OP_TYPE.c_str(),
              "BatchToSpaceND has input of crops which first value is not 0 by using NCHW, graph not changed.");
      return NOT_CHANGED;
    }
  }

  ge::NodePtr fusion_node = nullptr;
  // const to attr
  Status ret = PatternFusionUtil::ConstToAttrWithNode(graph, fusedNode, fusionOpType, attrInfos, fusion_node);
  if (ret != SUCCESS) {
    OP_LOGI(FUSED_OP_TYPE.c_str(), "BatchToSpaceND has input which is not a constant, graph not changed.");
    return NOT_CHANGED;
  }
  fusionNodes.push_back(fusion_node);
  OP_LOGI(FUSED_OP_TYPE.c_str(), "BatchToSpaceNDD fusion SUCCESSS!");
  return SUCCESS;
}
REGISTER_PASS("ConstToAttrBatchToSpaceNdFusion", BUILT_IN_GRAPH_PASS, ConstToAttrBatchToSpaceNdPass);
}  // namespace fe
