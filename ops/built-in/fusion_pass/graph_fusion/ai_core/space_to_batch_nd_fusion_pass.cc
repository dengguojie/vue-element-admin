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
 * \file space_to_batch_nd_fusion_pass.cc
 * \brief SpaceToBatchND fusion pass(SpaceToBatchND --> SpaceToBatchNDD)
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
#include "tbe_ops_pass_util.h"

using namespace ge;
namespace fe {
static const char* FUSED_NODE = "SpaceToBatchND";
static const string PATTERN_FUSED_NODE = "SpaceToBatchND";

static void CalcData(const Tensor& data, const DataType& dtype, vector<int64_t>& constVec) {
  const uint8_t* const_data = data.GetData();
  if (const_data == nullptr) {
    return;
  }
  size_t size;
  if (dtype == DT_INT32) {
    size = data.GetSize() / sizeof(int32_t);
    for (size_t i = 0; i < size; ++i) {
      constVec.push_back(*((int32_t*)const_data + i));
    }
  } else {
    size = data.GetSize() / sizeof(int64_t);
    for (size_t i = 0; i < size; ++i) {
      constVec.push_back(*((int64_t*)const_data + i));
    }
  }
}

vector<FusionPattern*> ConstToAttrSpaceToBatchNdPass::DefinePatterns() {
  vector<FusionPattern*> patterns;
  FusionPattern* pattern = new (nothrow) FusionPattern("ConstToAttrSpaceToBatchNdPass");
  FUSION_PASS_CHECK(pattern == nullptr, OP_LOGE(FUSED_OP_TYPE.c_str(), "new a pattern object failed."),
                    return patterns);
  pattern->AddOpDesc(PATTERN_FUSED_NODE, {FUSED_NODE}).SetOutput(PATTERN_FUSED_NODE);
  patterns.push_back(pattern);
  return patterns;
}

// vector<NodePtr> &fusionNodes: Store fusion nodes,
//       including newly added nodes and fused but not deleted nodes
Status ConstToAttrSpaceToBatchNdPass::Fusion(ComputeGraph& graph, Mapping& mapping, vector<NodePtr>& fusionNodes) {
  OP_LOGI(FUSED_OP_TYPE.c_str(), "SpaceToBatchND fusion in!");
  // build attr infos
  string fusion_optype = "SpaceToBatchNDD";
  vector<PassAttrInfo> attr_infos;
  PassAttrInfo block_shape = {1, "block_shape", "SetListInt"};
  attr_infos.push_back(block_shape);
  PassAttrInfo paddings = {2, "paddings", "SetListInt"};
  attr_infos.push_back(paddings);

  // get node
  NodePtr space_node = GetNodeFromMapping(PATTERN_FUSED_NODE, mapping);
  FUSION_PASS_CHECK(space_node == nullptr, OP_LOGE(FUSED_OP_TYPE.c_str(), "space_node is null, fusion failed."),
                    return PARAM_INVALID);

  // build a fusion node op desc
  OpDescPtr fusion_desc = PatternFusionUtil::GetFusionOpDesc(space_node, fusion_optype, attr_infos);
  FUSION_PASS_CHECK(fusion_desc == nullptr, OP_LOGI(FUSED_OP_TYPE.c_str(), "fusion_desc is null."), return NOT_CHANGED);
  // check op support
  FUSION_PASS_CHECK(!CheckOpSupported(fusion_desc), OP_LOGI(FUSED_OP_TYPE.c_str(), "SpaceToBatchNDD not supported."),
                    return NOT_CHANGED);

  // get op
  Operator sapce_op = OpDescUtils::CreateOperatorFromNode(space_node);

  // get const data
  Tensor block_tensor;
  if (sapce_op.GetInputConstData("block_shape", block_tensor) != GRAPH_SUCCESS) {
    OP_LOGI(FUSED_OP_TYPE.c_str(),
            "SpaceToBatchND has input of block_shape which is not a constant, graph not changed.");
    return NOT_CHANGED;
  }
  DataType block_type = sapce_op.GetInputDesc("block_shape").GetDataType();
  vector<int64_t> block_vec;
  CalcData(block_tensor, block_type, block_vec);

  Tensor paddings_tensor;
  if (sapce_op.GetInputConstData("paddings", paddings_tensor) != GRAPH_SUCCESS) {
    OP_LOGI(FUSED_OP_TYPE.c_str(), "SpaceToBatchND has input of paddings which is not a constant, graph not changed.");
    return NOT_CHANGED;
  }
  DataType paddings_type = sapce_op.GetInputDesc("paddings").GetDataType();
  vector<int64_t> paddings_vec;
  CalcData(paddings_tensor, paddings_type, paddings_vec);

  // get input
  TensorDesc input_desc = sapce_op.GetInputDesc("x");
  Format input_format = input_desc.GetFormat();
  Shape input_shape = input_desc.GetShape();
  size_t input_dim_num = input_shape.GetDimNum();

  // check dynamic shape
  FUSION_PASS_CHECK(IsUnknownShape(input_shape.GetDims()), OP_LOGI(FUSED_OP_TYPE.c_str(), "SpaceToBatchND is dynamic."),
                    return NOT_CHANGED);

  // check input
  if ((input_format != FORMAT_NHWC) && (input_format != FORMAT_NCHW) && (input_format != FORMAT_NDHWC) &&
      (input_format != FORMAT_NCDHW)) {
    OP_LOGI(FUSED_OP_TYPE.c_str(),
            "SpaceToBatchND has input which format is not NHWC or NCHW or NDHWC or NCDHW, graph not changed, got "
            "format is %d.",
            input_format);
    return NOT_CHANGED;
  }

  if (input_format == FORMAT_NHWC) {
    if ((input_dim_num != 4) || (block_vec.size() != 2) || (paddings_vec.size() != 4)) {
      if ((input_dim_num != 3) || (block_vec.size() != 1) || (paddings_vec.size() != 2)) {
        OP_LOGI(FUSED_OP_TYPE.c_str(),
                "SpaceToBatchND has input with format 'NHWC' which does not meet the rules, graph not changed, got "
                "input dim num is %d, block size is %d, pads size is %d.",
                input_dim_num, block_vec.size(), paddings_vec.size());
        return NOT_CHANGED;
      }
    }
  }

  if ((input_format == FORMAT_NCHW) && ((input_dim_num != 4) || (block_vec.size() != 3) || (paddings_vec.size() != 6) ||
                                        (block_vec[0] != 1) || (paddings_vec[0] != 0) || (paddings_vec[1] != 0))) {
    OP_LOGI(FUSED_OP_TYPE.c_str(),
            "SpaceToBatchND has input with format 'NCHW' which does not meet the rules, graph not changed, got input "
            "dim num is %d, block size is %d, pads size is %d, block[0] is %d, pads[0] is %d, pads[1] is %d.",
            input_dim_num, block_vec.size(), paddings_vec.size(), block_vec[0], paddings_vec[0], paddings_vec[1]);
    return NOT_CHANGED;
  }

  if ((input_format == FORMAT_NDHWC) &&
      ((input_dim_num != 5) || (block_vec.size() != 3) || (paddings_vec.size() != 6))) {
    OP_LOGI(FUSED_OP_TYPE.c_str(),
            "SpaceToBatchND has input with format 'NDHWC' which does not meet the rules, graph not changed, got "
            "input dim num is %d, block size is %d, pads size is %d.",
            input_dim_num, block_vec.size(), paddings_vec.size());
    return NOT_CHANGED;
  }

  if ((input_format == FORMAT_NCDHW) &&
      ((input_dim_num != 5) || (block_vec.size() != 4) || (paddings_vec.size() != 8) || (block_vec[0] != 1) ||
       (paddings_vec[0] != 0) || (paddings_vec[1] != 0))) {
    OP_LOGI(FUSED_OP_TYPE.c_str(),
            "SpaceToBatchND has input with format 'NCDHW' which does not meet the rules, graph not changed, got input "
            "dim num is %d, block size is %d, pads size is %d, block[0] is %d, pads[0] is %d, pads[1] is %d.",
            input_dim_num, block_vec.size(), paddings_vec.size(), block_vec[0], paddings_vec[0], paddings_vec[1]);
    return NOT_CHANGED;
  }

  // const to attr
  NodePtr fusion_node = nullptr;
  Status ret = PatternFusionUtil::ConstToAttrWithNode(graph, space_node, fusion_optype, attr_infos, fusion_node);
  if (ret != SUCCESS) {
    OP_LOGI(FUSED_OP_TYPE.c_str(), "SpaceToBatchND execute const to attr failed, graph not changed.");
    return NOT_CHANGED;
  }
  fusionNodes.push_back(fusion_node);
  OP_LOGI(FUSED_OP_TYPE.c_str(), "SpaceToBatchND fusion success!");
  return SUCCESS;
}
REGISTER_PASS("ConstToAttrSpaceToBatchNdPass", BUILT_IN_GRAPH_PASS, ConstToAttrSpaceToBatchNdPass);
}  // namespace fe
