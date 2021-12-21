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
 * \file conv2d_dequant_add_mul_quant_pass.cpp
 * \brief tbe conv2d + add + mul + quant ops fusion pattern
 */
#include <string>
#include <memory>
#include "op_log.h"
#include "conv2d_reluv2_pass.h"
#include "graph_optimizer/buffer_fusion/buffer_fusion_pass_registry.h"
#include "graph/utils/attr_utils.h"
#include "pattern_fusion_util.h"
#include "conv2d_slice_info_cal_base.h"

namespace fe {
using std::string;

static string g_kPatternConv = "conv2d";
static string g_kPatternReluv2 = "reluv2";

/*
 * @brief:  define conv2d op fusion pattern
 *
 * pattern configuration limit:
 * 1. total min value must be 1 for all head candidated desc.
 * 2. any head candidated desc max value must be 1.
 * 3. output desc can not be itself.
 *
 *    conv2d --> reluv2  --> relu_value
 *                     \ --> mask_value
 *
 * @return BufferFusionPattern: return all valid patterns.
 */
vector<BufferFusionPattern*> TbeConv2DReluv2Pass::DefinePatterns() {
  vector<BufferFusionPattern*> patterns;
  string passName = "TbeConv2DReluv2Fusion";
  BufferFusionPattern *pattern = new (std::nothrow) BufferFusionPattern(passName);
  FUSION_PASS_CHECK((pattern == nullptr), OP_LOGE(fused_op_type_.c_str(), "create new pattern failed."),
                    return patterns);
  OP_LOGD(fused_op_type_.c_str(), "Start to define %s pass pattern.", passName.c_str());
  pattern->AddOpDesc(g_kPatternConv, {OP_PATTERN_CONV}, TBE_PATTERN_NUM_DEFAULT, TBE_PATTERN_NUM_DEFAULT, TBE_PATTERN_GROUPID_INVALID, IGNORE_SHAPE_TYPE)
      .AddOpDesc(g_kPatternReluv2, {OP_PATTERN_ELEMWISE}, TBE_PATTERN_NUM_DEFAULT, TBE_PATTERN_NUM_DEFAULT, TBE_PATTERN_GROUPID_INVALID, IGNORE_SHAPE_TYPE)
      .SetHead({g_kPatternConv})
      .SetOutputs(g_kPatternConv, {g_kPatternReluv2});
  patterns.push_back(pattern);
  OP_LOGD(fused_op_type_.c_str(), "End to define %s pass pattern.", passName.c_str());

  return patterns;
}

/*
 * @brief: parse nodes matched in mapping and call DoFusion
 * @param [in] graph: original graph
 * @param [out] mapping: nodes matched by pattern
 * @return bool: fusion status ok or not.
 */
Status TbeConv2DReluv2Pass::GetFusionNodes(const BufferFusionMapping& mapping, vector<ge::NodePtr>& fusion_nodes) {
  OP_LOGD(fused_op_type_.c_str(), "Begin to do Conv2dRelu2Fusion!");
  if (mapping.size() != 2) {
    OP_LOGD(fused_op_type_.c_str(), "mapping size should be 2");
  }
  auto conv2d_node = GetMatchedNodesByDescName(g_kPatternConv, mapping);
  auto reluv2_node = GetMatchedNodesByDescName(g_kPatternReluv2, mapping);
  FUSION_PASS_CHECK(conv2d_node.size() != 1,
                    OP_LOGD(fused_op_type_.c_str(), "conv2d node should be matched only once!"),
                    return SUCCESS);
  FUSION_PASS_CHECK(conv2d_node[0]->GetOpDesc() == nullptr,
                    OP_LOGD(fused_op_type_.c_str(), "get desc failed"),
                    return SUCCESS);
  FUSION_PASS_CHECK(reluv2_node.size() != 1,
                    OP_LOGD(fused_op_type_.c_str(), "reluv2 node should match only once!"),
                    return SUCCESS);
  FUSION_PASS_CHECK(conv2d_node[0]->GetOpDesc()->GetOutputDesc(0).GetDataType() != DT_FLOAT16,
                    OP_LOGD(fused_op_type_.c_str(), "conv2d node only supports float16!"),
                    return SUCCESS);
  FUSION_PASS_CHECK((reluv2_node[0]->GetAllOutAnchors().size() != 2 || reluv2_node[0]->GetType() != "ReluV2"),
                    OP_LOGD(fused_op_type_.c_str(), "reluv2 should have two output anchors"),
                    return SUCCESS);
  fusion_nodes = GetMatchedNodes(mapping);

  OP_LOGD(fused_op_type_.c_str(), "End to do Conv2DReluv2Fusion!");
  return SUCCESS;
}

Status TbeConv2DReluv2Pass::CalcFusionOpSliceInfo(vector<ge::NodePtr> &fusion_nodes, OpCalcInfo &op_slice_info)
{
  OP_LOGD(fused_op_type_.c_str(), "start calc slice info.");
  std::unique_ptr<ConvSliceInfoCalBase> pConvSliceInfoCal = nullptr;
  pConvSliceInfoCal.reset(new (std::nothrow) ConvSliceInfoCalBase());
  CONV_RET_IF_SMART_PTR_IS_NULL(pConvSliceInfoCal);
  Status ret = pConvSliceInfoCal->ConvCalcFusionOpSliceInfo(fusion_nodes, op_slice_info, fused_op_type_);
  FUSION_PASS_CHECK(ret != SUCCESS, OP_LOGW(fused_op_type_.c_str(), "calc fusion op slice info failed."), return FAILED);
  OP_LOGD(fused_op_type_.c_str(), "end calc slice info.");
  return SUCCESS;
}

REGISTER_BUFFER_FUSION_PASS("TbeConv2DReluv2Pass", BUILT_IN_AI_CORE_BUFFER_FUSION_PASS, TbeConv2DReluv2Pass);
}  // namespace fe
