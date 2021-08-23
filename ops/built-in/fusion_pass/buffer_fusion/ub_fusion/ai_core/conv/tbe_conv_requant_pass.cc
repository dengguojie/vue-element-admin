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

#include <memory>
#include "tbe_conv_requant_pass.h"
#include "op_log.h"
#include "pattern_fusion_util.h"
#include "graph_optimizer/buffer_fusion/buffer_fusion_pass_registry.h"
#include "conv2d_slice_info_cal_base.h"

namespace fe {
using std::vector;
static const string PATTERN_CONV = "convolution";
static const string PATTERN_REQUANT = "requant";
static const string PATTERN_OTHER_INPUT = "otherInput";
static const string PATTERN_STRIDEDWRITE = "strided_write";

vector<BufferFusionPattern*> ConvRequantFusionPass::DefinePatterns() {
  vector<BufferFusionPattern*> patterns;

  string pattern_name = "TbeConvRequantFusion";
  BufferFusionPattern* pattern = new (std::nothrow) BufferFusionPattern(pattern_name);
  FUSION_PASS_CHECK((pattern == nullptr), OP_LOGE(fused_op_type_.c_str(), "new an object failed."), return patterns);
  OP_LOGD(fused_op_type_.c_str(), "Start to define %s pass pattern.", pattern_name.c_str());
  // define pattern rules
  pattern->AddOpDesc(PATTERN_CONV, {OP_PATTERN_CONV}, TBE_PATTERN_NUM_DEFAULT, TBE_PATTERN_NUM_DEFAULT, TBE_PATTERN_GROUPID_INVALID, IGNORE_SHAPE_TYPE)
          .AddOpDesc(PATTERN_REQUANT, {OP_PATTERN_REQUANT}, TBE_PATTERN_NUM_DEFAULT, TBE_PATTERN_NUM_DEFAULT, TBE_PATTERN_GROUPID_INVALID, IGNORE_SHAPE_TYPE)
          .AddOpDesc(PATTERN_OTHER_INPUT, {TBE_PATTERN_INPUT_NODE}, TBE_PATTERN_NUM_DEFAULT, TBE_PATTERN_NUM_DEFAULT, TBE_PATTERN_GROUPID_INVALID, IGNORE_SHAPE_TYPE)
          .AddOpDesc(PATTERN_STRIDEDWRITE, {OP_PATTERN_STRIDED_WRITE}, TBE_PATTERN_NUM_NONE, TBE_PATTERN_NUM_DEFAULT, TBE_PATTERN_GROUPID_INVALID, IGNORE_SHAPE_TYPE)
          .SetHead({PATTERN_CONV})
          .SetOutputs(PATTERN_CONV, {PATTERN_REQUANT})
          .SetOutputs(PATTERN_REQUANT, {PATTERN_STRIDEDWRITE})
          .SetOutputs(PATTERN_OTHER_INPUT, {PATTERN_REQUANT});
  patterns.push_back(pattern);
  OP_LOGD(fused_op_type_.c_str(), "End to define %s pass pattern.", pattern_name.c_str());
  return patterns;
}

Status ConvRequantFusionPass::CalcFusionOpSliceInfo(vector<ge::NodePtr> &fusion_nodes, OpCalcInfo &op_slice_info)
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

REGISTER_BUFFER_FUSION_PASS("TbeConvRequantFusionPass", BUILT_IN_AI_CORE_BUFFER_FUSION_PASS, ConvRequantFusionPass);
}  // namespace fe