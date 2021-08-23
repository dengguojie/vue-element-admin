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
 * \file depthwise_df_fusion_pass.cpp
 * \brief depthwise_df_fusion_pass
 */
#include "depthwise_df_fusion_pass.h"

#include <iostream>
#include <vector>
#include <string>
#include <map>

#include "quant_host_cpu_op_common.h"
#include "op_log.h"
#include "fp16_t.hpp"
#include "graph/debug/ge_attr_define.h"
#include "graph/types.h"
#include "graph/utils/attr_utils.h"
#include "graph/utils/graph_utils.h"
#include "graph/utils/node_utils.h"
#include "graph/utils/op_desc_utils.h"
#include "graph_optimizer/graph_fusion/fusion_pass_manager/fusion_pass_registry.h"
#include "pattern_fusion_util.h"
#include "securec.h"

using namespace std;
using namespace ge;

namespace fe {
static const string PATTERN_DEPTHWISE_DF = "DepthwiseConv2DBackpropInputD";
static const char* DEPTHWISE = "DepthwiseConv2DBackpropInputD";
static const char* DEPTHWISE_DYN = "DepthwiseConv2DBackpropInput";
const int MAX_DIM_NUM = 4;

vector<FusionPattern*> DepthwiseDfFusionPass::DefinePatterns() {
  vector<FusionPattern*> patterns;

  // define DepthwiseDfFusion
  FusionPattern* pattern = new (std::nothrow) FusionPattern("DepthwiseDfFusionPass");
  FUSION_PASS_CHECK(pattern == nullptr, OP_LOGE(FUSED_OP_TYPE.c_str(), "new a pattern object failed."),
                    return patterns);
  // define origin graph
  pattern->AddOpDesc(PATTERN_DEPTHWISE_DF, {DEPTHWISE, DEPTHWISE_DYN}).SetOutput(PATTERN_DEPTHWISE_DF);
  patterns.push_back(pattern);

  return patterns;
}

Status DepthwiseDfFusionPass::SwapNumChnImpl(GeTensorDesc& tensor_desc) {
  FUSION_PASS_CHECK (
    tensor_desc.GetShape().GetDimNum() != MAX_DIM_NUM,
    OP_LOGE(FUSED_OP_TYPE.c_str(), "dim count not illegal, need:4 real:%d", tensor_desc.GetShape().GetDimNum()),
    return PARAM_INVALID);
  ge::Format origin_format = tensor_desc.GetOriginFormat();
  ge::GeShape tensor_shape = tensor_desc.GetShape();
  vector<int64_t> dim_info = tensor_shape.GetDims();
  int64_t n = 0;
  int64_t c = 0;
  int64_t h = 0;
  int64_t w = 0;
  if (dim_info.size() == 4) {
    if (origin_format == FORMAT_HWCN) {
      OP_LOGD(FUSED_OP_TYPE.c_str(), "in FORMAT_HWCN, before swap C, N, H, W: [%d, %d, %d, %d]", (int)dim_info[2],
      (int)dim_info[3], (int)dim_info[0], (int)dim_info[1]);
      n = dim_info[3];
      h = dim_info[0];
      w = dim_info[1];
      c = dim_info[2];
      tensor_desc.SetShape(ge::GeShape({h, w, 1, c*n}));
      tensor_desc.SetOriginShape(ge::GeShape({h, w, 1, c*n}));
    }
  } else {
    OP_LOGW(FUSED_OP_TYPE.c_str(), "dim_info is not right, please check!");
    return NOT_CHANGED;
  }
  return SUCCESS;
}

Status DepthwiseDfFusionPass::SwapNumChn(OpDescPtr& op_desc, bool b_input, uint32_t index, bool both) {
  bool flag = false;
  ge::GeTensorDesc tensor_desc;
  graphStatus ret_res;
  auto get_all_input_candidate_desc = op_desc->GetAllInputsDesc();

  if (both) {
    vector<int64_t> dim_info;
    for (auto iter=get_all_input_candidate_desc.begin(); iter!=get_all_input_candidate_desc.end(); ++iter) {
      tensor_desc = *iter;
      if (tensor_desc.GetShape().GetDimNum() == MAX_DIM_NUM) {
        FUSION_PASS_CHECK(SwapNumChnImpl(tensor_desc) != SUCCESS,
                          OP_LOGE(FUSED_OP_TYPE.c_str(), "Conv parent const node out 0 change nc failed"), return FAILED);
        ret_res = op_desc->UpdateInputDesc(index, tensor_desc);
        dim_info = tensor_desc.GetShape().GetDims();
        OP_LOGD(FUSED_OP_TYPE.c_str(), "after swap: [%d, %d, %d, %d]", (int)dim_info[0],
                   (int)dim_info[1], (int)dim_info[2], (int)dim_info[3]);
        flag = true;
      }
    }
    if (op_desc->GetOutputDesc(0).GetShape().GetDimNum() == MAX_DIM_NUM) {
      tensor_desc = op_desc->GetOutputDesc(0);
      FUSION_PASS_CHECK(SwapNumChnImpl(tensor_desc) != SUCCESS,
                        OP_LOGE(FUSED_OP_TYPE.c_str(), "Conv parent const node out 0 change nc failed"), return FAILED);
      ret_res = op_desc->UpdateOutputDesc(index, tensor_desc);
      dim_info = tensor_desc.GetShape().GetDims();
      OP_LOGD(FUSED_OP_TYPE.c_str(), "after swap: [%d, %d, %d, %d]", (int)dim_info[0],
              (int)dim_info[1], (int)dim_info[2], (int)dim_info[3]);
      flag = true;
    }
  } else {
    if (b_input) {
      tensor_desc = op_desc->GetInputDesc(index);
      FUSION_PASS_CHECK(SwapNumChnImpl(tensor_desc) != SUCCESS,
                        OP_LOGE(FUSED_OP_TYPE.c_str(), "Conv parent const node out 0 change nc failed"), return FAILED);
      ret_res = op_desc->UpdateInputDesc(index, tensor_desc);

      vector<int64_t> dim_info1 = tensor_desc.GetShape().GetDims();
      OP_LOGD(FUSED_OP_TYPE.c_str(), "after swap input: [%d, %d, %d, %d]", (int)dim_info1[0],
              (int)dim_info1[1], (int)dim_info1[2], (int)dim_info1[3]);
      flag = true;
    } else {
      tensor_desc = op_desc->GetOutputDesc(index);
      FUSION_PASS_CHECK(SwapNumChnImpl(tensor_desc) != SUCCESS,
                        OP_LOGE(FUSED_OP_TYPE.c_str(), "Conv parent const node out 0 change nc failed"), return FAILED);
      ret_res = op_desc->UpdateOutputDesc(index, tensor_desc);

      vector<int64_t> dim_info1 = tensor_desc.GetShape().GetDims();
      OP_LOGD(FUSED_OP_TYPE.c_str(), "after swap output: [%d, %d, %d, %d]", (int)dim_info1[0],
              (int)dim_info1[1], (int)dim_info1[2], (int)dim_info1[3]);
      flag = true;
    }
  }
  if (flag) {
    FUSION_PASS_CHECK(ret_res != ge::GRAPH_SUCCESS, OP_LOGE(FUSED_OP_TYPE.c_str(), 
                      "Update matmul variable failed"), return PARAM_INVALID);
  }
  return SUCCESS;
}

Status DepthwiseDfFusionPass::Fusion(ge::ComputeGraph& graph, Mapping& mapping, vector<ge::NodePtr>& fusion_nodes) {
  OP_LOGI("Enter DepthwiseDfFusionPass");
  ge::NodePtr depthwise_node = GetNodeFromMapping(PATTERN_DEPTHWISE_DF, mapping);
  OpDescPtr depthwise_desc = depthwise_node->GetOpDesc();
  OP_LOGD(depthwise_desc->GetName().c_str(), "dealing with df fusion");

  uint32_t index = 0;
  if (depthwise_desc->GetType() == DEPTHWISE_DYN) {
    index = 1;
  }
  FUSION_PASS_CHECK(SwapNumChn(depthwise_desc, true, index, false) != SUCCESS,
                    OP_LOGE(FUSED_OP_TYPE.c_str(), "Conv parent const node out 0 change nc failed"), return FAILED);
  ge::NodePtr filter_node = depthwise_node->GetInDataNodes().at(index);
  OpDescPtr filter_desc = filter_node->GetOpDesc();

  FUSION_PASS_CHECK(SwapNumChn(filter_desc, true, 0, true) != SUCCESS,
                    OP_LOGE(FUSED_OP_TYPE.c_str(), "Conv parent const node out 0 change nc failed"), return FAILED);
  OP_LOGI("Leave DepthwiseDfFusionPass");
  return SUCCESS;
}
REGISTER_PASS("DepthwiseDfFusionPass", BUILT_IN_GRAPH_PASS, DepthwiseDfFusionPass);
}
