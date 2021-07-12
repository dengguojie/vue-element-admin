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
 * \file a_depthwise_fusion_pass.cpp
 * \brief a_depthwise_fusion_pass
 */
#include "a_depthwise_fusion_pass.h"

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

namespace fe{
  static const string PATTERN_DEPTHWISE = "DepthwiseConv2D";
  static const char *DEPTHWISE = "DepthwiseConv2D";
  const int MAX_DIM_NUM = 4;
  const int64_t ALREADY_CHANGED_C = 1;

  vector<FusionPattern *> DepthwiseFusionPass::DefinePatterns() {
    vector<FusionPattern *> patterns;

    // define AvgPoolFusion
    FusionPattern *pattern = new (std::nothrow) FusionPattern("ADepthwiseConv2D");
    FUSION_PASS_CHECK(pattern == nullptr, OP_LOGE(FUSED_OP_TYPE.c_str(), "new a pattern object failed."),
                      return patterns);
    // define origin graph
    pattern->AddOpDesc(PATTERN_DEPTHWISE, {DEPTHWISE}).SetOutput(PATTERN_DEPTHWISE);
    patterns.push_back(pattern);

    return patterns;
  }

  Status DepthwiseFusionPass::SwapNumChnImpl(GeTensorDesc &tensor_desc) {
    FUSION_PASS_CHECK(
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
      if (origin_format == FORMAT_NHWC) {
        OP_LOGI(FUSED_OP_TYPE.c_str(), "in FORMAT_NHWC, before swap N, H, W, C: [%d, %d, %d, %d]", (int)dim_info[3],
                (int)dim_info[0], (int)dim_info[1], (int)dim_info[2]);
        return NOT_CHANGED;
      } else if (origin_format == FORMAT_HWCN) {
        OP_LOGI(FUSED_OP_TYPE.c_str(), "in FORMAT_HWCN, before swap H, W, C, N: [%d, %d, %d, %d]", (int)dim_info[0],
                (int)dim_info[1], (int)dim_info[2], (int)dim_info[3]);
        n = dim_info[3];
        h = dim_info[0];
        w = dim_info[1];
        c = dim_info[2];
        if (c == ALREADY_CHANGED_C) {
          OP_LOGI(FUSED_OP_TYPE.c_str(), "in c == ALREADY_CHANGED_C, return NOT_CHANGED");
          return NOT_CHANGED;
        }
        tensor_desc.SetShape(ge::GeShape({h, w, 1, n * c}));
        tensor_desc.SetOriginShape(ge::GeShape({h, w, 1, n * c}));
      }
    } else {
      OP_LOGW(FUSED_OP_TYPE.c_str(), "dim_info is not right, please check!");
      return FAILED;
    }
    return SUCCESS;
  }

  Status DepthwiseFusionPass::SwapNumChn(OpDescPtr &op_desc, bool b_input, int index) {
    bool flag = false;
    ge::GeTensorDesc tensor_desc;
    graphStatus ret_res;
    auto get_all_input_candidate_desc = op_desc->GetAllInputsDesc();
    auto get_all_output_candidate_desc = op_desc->GetAllOutputsDesc();
    ge::Format origin_format;
    vector<int64_t> dim_info;
    if (ge::AttrUtils::HasAttr(op_desc, "_has_been_changed")) {
      OP_LOGI(op_desc->GetName().c_str(), "has_been_changed");
      return NOT_CHANGED;
    }

    if (b_input) {
      if (index == -1) {
        if (!get_all_input_candidate_desc.empty()) {
          OP_LOGI("in swap all input num channel and get_all_input_candidate_desc not empty");
          int count = 0;
          for (auto iter = get_all_input_candidate_desc.begin(); iter != get_all_input_candidate_desc.end(); ++iter) {
            tensor_desc = *iter;
            if (tensor_desc.GetShape().GetDimNum() == MAX_DIM_NUM) {
              auto result = SwapNumChnImpl(tensor_desc);
              if (result == NOT_CHANGED) {
                return NOT_CHANGED;
              }
              FUSION_PASS_CHECK(result != SUCCESS,
                                OP_LOGE(op_desc->GetName().c_str(), "Conv parent const node out 0 change nc failed"), return FAILED);
              ret_res = op_desc->UpdateInputDesc(count, tensor_desc);
              origin_format = tensor_desc.GetOriginFormat();
              dim_info = tensor_desc.GetShape().GetDims();
              flag = true;
            }
            count += 1;
          }
          if (!flag) {
            OP_LOGI(op_desc->GetName().c_str(), "!flag, return NOT_CHANGED");
            return NOT_CHANGED;
          }
        } else {
          OP_LOGI("in swap all input num channel and get_all_input_candidate_desc is empty");
          return NOT_CHANGED;
        }
      } else {
        OP_LOGI("in swap input special num channel");
        tensor_desc = op_desc->GetInputDesc(index);
        auto result = SwapNumChnImpl(tensor_desc);
        FUSION_PASS_CHECK(result == FAILED,
                          OP_LOGE(op_desc->GetName().c_str(), "Conv parent const node out 0 change nc failed"), return FAILED);
        ret_res = op_desc->UpdateInputDesc(index, tensor_desc);

        ge::Format origin_format = tensor_desc.GetOriginFormat();
        vector<int64_t> dim_info = tensor_desc.GetShape().GetDims();
        OP_LOGI(op_desc->GetName().c_str(), "after swap: [%d, %d, %d, %d]", (int)dim_info[0],
                (int)dim_info[1], (int)dim_info[2], (int)dim_info[3]);
        flag = true;
      }
    } else {
      if (index == -1) {
        if (!get_all_output_candidate_desc.empty()) {
          OP_LOGI("in swap all output num channel and get_all_input_candidate_desc not empty");
          int count = 0;
          for (auto iter = get_all_output_candidate_desc.begin(); iter != get_all_output_candidate_desc.end(); ++iter) {
            tensor_desc = *iter;
            if (tensor_desc.GetShape().GetDimNum() == MAX_DIM_NUM) {
              auto result = SwapNumChnImpl(tensor_desc);
              if (result == NOT_CHANGED) {
                return NOT_CHANGED;
              }
              FUSION_PASS_CHECK(result != SUCCESS,
                                OP_LOGE(op_desc->GetName().c_str(), "Conv parent const node out 0 change nc failed"), return FAILED);
              ret_res = op_desc->UpdateOutputDesc(count, tensor_desc);
              origin_format = tensor_desc.GetOriginFormat();
              dim_info = tensor_desc.GetShape().GetDims();
              flag = true;
            }
            count += 1;
          }
          if (!flag) {
            return NOT_CHANGED;
          }
        } else {
          OP_LOGI("in swap all output num channel and get_all_input_candidate_desc is empty");
          return NOT_CHANGED;
        }
      } else {
        OP_LOGI("in swap output special num channel");
        tensor_desc = op_desc->GetOutputDesc(index);
        auto result = SwapNumChnImpl(tensor_desc);
        FUSION_PASS_CHECK(result == FAILED,
                          OP_LOGE(op_desc->GetName().c_str(), "Conv parent const node out 0 change nc failed"), return FAILED);
        ret_res = op_desc->UpdateOutputDesc(index, tensor_desc);

        ge::Format origin_format = tensor_desc.GetOriginFormat();
        vector<int64_t> dim_info = tensor_desc.GetShape().GetDims();
        flag = true;
      }
    }
    if (flag) {
      FUSION_PASS_CHECK(ret_res != ge::GRAPH_SUCCESS, OP_LOGE(op_desc->GetName().c_str(), "Update matmul variable failed"),
                        return PARAM_INVALID);
    }
    return SUCCESS;
  }

  Status DepthwiseFusionPass::GetNodeInSameLevel(NodePtr &tmp_node, bool b_input, int index, int level_left) {
    if ((tmp_node) && (tmp_node->GetOpDesc()) && level_left > 0) {
      OpDescPtr tmp_node_desc = tmp_node->GetOpDesc();
      if (!tmp_node->GetInDataNodes().empty()) {
        auto result = SwapNumChn(tmp_node_desc, true, -1);
        FUSION_PASS_CHECK(result == FAILED,
                          OP_LOGE(tmp_node_desc->GetName().c_str(), "Conv parent const node out 0 change nc failed"), return FAILED);
        if (result == NOT_CHANGED) {
          OP_LOGI(tmp_node_desc->GetName().c_str(), "input return is NOT_CHANGED");
          return NOT_CHANGED;
        } else {
          auto get_all_input_candidate_node = tmp_node->GetInDataNodes();
          OP_LOGI(tmp_node_desc->GetName().c_str(), "looking for upper node");
          ge::NodePtr upperNode;
          for (auto iter = get_all_input_candidate_node.begin(); iter != get_all_input_candidate_node.end(); ++iter) {
            upperNode = *iter;
            auto upper_node_result = GetNodeInSameLevel(upperNode, b_input, index, level_left-1);
            FUSION_PASS_CHECK(upper_node_result == FAILED,
                              OP_LOGE(tmp_node_desc->GetName().c_str(), "Conv parent const node out 0 change nc failed"), return FAILED);
          }
        }
      }
      if (!tmp_node_desc->GetAllOutputsDesc().empty()) {
        OP_LOGI(tmp_node_desc->GetName().c_str(), "begin change output, index: %d", index);
        auto result = SwapNumChn(tmp_node_desc, false, -1);
        FUSION_PASS_CHECK(result == FAILED,
                          OP_LOGE(tmp_node_desc->GetName().c_str(), "Conv parent const node out 0 change nc failed"), return FAILED);
        if (result == NOT_CHANGED) {
          OP_LOGI(tmp_node_desc->GetName().c_str(), "output return is NOT_CHANGED");
          return NOT_CHANGED;
        } else {
          auto get_all_output_candidate_node = tmp_node->GetOutDataNodes();
          OP_LOGI(tmp_node_desc->GetName().c_str(), "looking for lower node");
          ge::NodePtr lower_node;
          for (auto iter = get_all_output_candidate_node.begin(); iter != get_all_output_candidate_node.end(); ++iter) {
            lower_node = *iter;
            auto lower_node_result = GetNodeInSameLevel(lower_node, b_input, index, level_left-1);
            FUSION_PASS_CHECK(lower_node_result == FAILED,
                              OP_LOGE(tmp_node_desc->GetName().c_str(), "Conv parent const node out 0 change nc failed"), return FAILED);
          }
        }
      }
      ge::AttrUtils::SetBool(tmp_node_desc, "_has_been_changed", true);
      return SUCCESS;
    }
  }

  Status DepthwiseFusionPass::Fusion(ge::ComputeGraph &graph, Mapping &mapping, vector<ge::NodePtr> &fusionNodes) {
    OP_LOGI("Enter DepthwiseFusionPass");
    ge::NodePtr depthwise_node = GetNodeFromMapping(PATTERN_DEPTHWISE, mapping);
    OpDescPtr depthwise_desc = depthwise_node->GetOpDesc();
    OP_LOGI(depthwise_desc->GetName().c_str(), "dealing with");
    auto result = SwapNumChn(depthwise_desc, true, 1);
    FUSION_PASS_CHECK(result == FAILED,
                      OP_LOGE(FUSED_OP_TYPE.c_str(), "Conv parent const node out 0 change nc failed"), return FAILED);
    ge::NodePtr filter_node = depthwise_node->GetInDataNodes().at(1);
    OpDescPtr filterDesc = filter_node->GetOpDesc();

    GetNodeInSameLevel(filter_node, true, -1, 12);
    OP_LOGI("Leave DepthwiseFusionPass");
    return SUCCESS;
  }
  REGISTER_PASS("ADepthwiseFusionPass", BUILT_IN_GRAPH_PASS, DepthwiseFusionPass);
} // namespace fe
