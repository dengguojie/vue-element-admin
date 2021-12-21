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
 * \file argmax_onehot_fusion_pass.cc
 * \brief
 */
#include "argmax_onehot_fusion_pass.h"

#include <map>
#include <vector>

#include "op_log.h"
#include "error_util.h"
#include "graph/utils/op_desc_utils.h"
#include "graph/utils/graph_utils.h"
#include "graph/utils/attr_utils.h"
#include "graph/debug/ge_attr_define.h"
#include "graph/utils/node_utils.h"
#include "graph_optimizer/graph_fusion/fusion_pass_manager/fusion_pass_registry.h"
#include "pattern_fusion_util.h"
#include "tbe_fusion_pass_util.h"
#include "securec.h"
#include "tbe_fusion_pass_util.h"
#include "op_common_util.h"

using namespace std;
using namespace ge;

static const char* ARGMAX_TYPE = "ArgMaxV2";
static const char* ARGMAX_PATTERN = ARGMAX_TYPE;
static const char* ONE_HOT_TYPE = "OneHot";
static const char* ONE_HOT_PATTERN = ONE_HOT_TYPE;
static const char* PASS_NAME = "ArgmaxOneHotFusionPass";
static const string DTYPE_ATTR_NAME = "dtype";

namespace fe {
vector<FusionPattern*> ArgmaxOneHotFusionPass::DefinePatterns() {
  vector<FusionPattern *> patterns;
  FusionPattern *pattern = new(std::nothrow) FusionPattern("ZOneHotDCastFusionPassPattern");
  FUSION_PASS_CHECK(pattern == nullptr,
                    VECTOR_FUSION_INNER_ERR_REPORT(PASS_NAME, "new a pattern object failed."),
                    return patterns);
  pattern->AddOpDesc(ARGMAX_PATTERN, {ARGMAX_TYPE})
      .AddOpDesc(ONE_HOT_PATTERN, {ONE_HOT_TYPE})
      .SetInputs(ONE_HOT_PATTERN, {ARGMAX_PATTERN})
      .SetOutput(ONE_HOT_PATTERN);
  patterns.push_back(pattern);

  return patterns;
}


Status ArgmaxOneHotFusionPass::Fusion(ge::ComputeGraph& graph, Mapping& mapping, vector<ge::NodePtr>& fusionNodes) {
  OP_LOGD(PASS_NAME, "%s is running.", PASS_NAME);
  auto one_hot_node = GetNodeFromMapping(ONE_HOT_PATTERN, mapping);
  FUSION_PASS_CHECK(!one_hot_node, OP_LOGD(PASS_NAME, "one_hot_node is null, fusion not change."),
                    return NOT_CHANGED);
  auto one_hot_input0_data_anchor = one_hot_node->GetInDataAnchor(0);
  FUSION_PASS_CHECK(!one_hot_input0_data_anchor,
                    OP_LOGD(PASS_NAME, "one_hot_node's input0 data anchor is null, fusion not change."),
                    return NOT_CHANGED);
  auto one_hot_input0_peer_out_data_anchor = one_hot_input0_data_anchor->GetPeerOutAnchor();
  FUSION_PASS_CHECK(!one_hot_input0_peer_out_data_anchor,
                    OP_LOGD(PASS_NAME, "one_hot_node's input0 peer out data anchor is null, fusion not change."),
                    return NOT_CHANGED);
  auto one_hot_input0_node = one_hot_input0_peer_out_data_anchor->GetOwnerNode();
  FUSION_PASS_CHECK(!one_hot_input0_node,
                    OP_LOGD(PASS_NAME, "one_hot_node's input0 data node is null, fusion not change."),
                    return NOT_CHANGED);
  OP_LOGD(PASS_NAME, "one_hot_node's input0 data node is %s, type is %s.",
          one_hot_input0_node->GetName().c_str(),
          one_hot_input0_node->GetType().c_str());
  if (one_hot_input0_node->GetType() != ARGMAX_TYPE) {
    OP_LOGD(PASS_NAME, "one_hot_node's input0 node is %s, but not argmax, fusion not change.",
            one_hot_input0_node->GetType().c_str());
    return NOT_CHANGED;
  }
  auto& argmax_node = one_hot_input0_node;
  auto argmax_op = OpDescUtils::CreateOperatorFromNode(argmax_node);
  if (argmax_op.GetOutputDesc(0).GetDataType() == ge::DT_INT32) {
    OP_LOGD(PASS_NAME, "data type of output[%s] is int32, fusion not change.", argmax_node->GetName().c_str());
    return NOT_CHANGED;
  }

  std::vector<int64_t> dimension;
  if (!TbeFusionPassUtil::GetConstIntData(argmax_op, "dimension", dimension) || dimension.empty()) {
    OP_LOGD(PASS_NAME, "Get dimension from %s failed, fusion not change.", argmax_node->GetName().c_str());
    return NOT_CHANGED;
  }

  auto argmax_first_input_shape = argmax_op.GetInputDesc(0).GetShape();
  if (argmax_first_input_shape.GetDims() == UNKNOWN_RANK) {
    OP_LOGD(PASS_NAME, "%s's shape is UNKNOWN_RANK, will not change.", argmax_node->GetName().c_str());
    return NOT_CHANGED;
  }

  if (dimension[0] < 0) {
    dimension[0] += static_cast<int64_t>(argmax_first_input_shape.GetDimNum());
  }

  OP_LOGD(PASS_NAME, "%s's dimension is %ld.", argmax_node->GetName().c_str(), dimension[0]);
  if (dimension[0] > argmax_first_input_shape.GetDimNum() || dimension[0] < 0) {
    OP_LOGD(PASS_NAME, "%s dimension[%ld] is invalid, fusion not change.", argmax_node->GetName().c_str(), dimension[0]);
    return NOT_CHANGED;
  }

  auto dim_value = static_cast<size_t>(dimension[0]);
  if (dim_value == ge::UNKNOWN_DIM) {
    OP_LOGD(PASS_NAME, "%s's shape is %s, dimension[%ld] is UNKNOWN_DIM, fusion not change.",
            argmax_node->GetName().c_str(),
            ops::to_string(argmax_first_input_shape.GetDims()).c_str(),
            dimension[0]);
    return NOT_CHANGED;
  }

  if (dim_value > std::numeric_limits<int32_t>::max()) {
    OP_LOGD(PASS_NAME, "%s's shape is %s, dimension[%ld] is larger than max value of int32(%ld), fusion not change.",
            argmax_node->GetName().c_str(),
            ops::to_string(argmax_first_input_shape.GetDims()).c_str(),
            dimension[0],
            std::numeric_limits<int32_t>::max());
    return NOT_CHANGED;
  }

  auto argmax_desc = argmax_node->GetOpDesc();
  FUSION_PASS_CHECK(!argmax_desc, OP_LOGD(PASS_NAME, "get argmax desc failed, fusion not change."),
                    return NOT_CHANGED);
  auto argmax_output_desc = argmax_desc->MutableOutputDesc(0);
  FUSION_PASS_CHECK(!argmax_output_desc, OP_LOGD(PASS_NAME, "get argmax output's desc failed, fusion not change."),
                    return NOT_CHANGED);
  auto one_hot_desc = one_hot_node->GetOpDesc();
  FUSION_PASS_CHECK(!one_hot_desc, OP_LOGD(PASS_NAME, "get one_hot desc failed, fusion not change."),
                    return NOT_CHANGED);
  auto one_hot_input0_desc = one_hot_desc->MutableInputDesc(0);
  FUSION_PASS_CHECK(!one_hot_input0_desc, OP_LOGD(PASS_NAME, "get one_hot input0's desc failed, fusion not change."),
                    return NOT_CHANGED);
  ge::AttrUtils::SetDataType(argmax_desc, DTYPE_ATTR_NAME, ge::DT_INT32);
  argmax_output_desc->SetDataType(ge::DT_INT32);
  argmax_output_desc->SetOriginDataType(ge::DT_INT32);
  argmax_node->UpdateOpDesc(argmax_desc);
  one_hot_input0_desc->SetDataType(ge::DT_INT32);
  one_hot_input0_desc->SetOriginDataType(ge::DT_INT32);
  one_hot_node->UpdateOpDesc(one_hot_desc);
  OP_LOGD(PASS_NAME, "%s run success.", PASS_NAME);
  return SUCCESS;
}

REGISTER_PASS("ArgmaxOneHotFusionPass", BUILT_IN_GRAPH_PASS, ArgmaxOneHotFusionPass);
}  // namespace fe

