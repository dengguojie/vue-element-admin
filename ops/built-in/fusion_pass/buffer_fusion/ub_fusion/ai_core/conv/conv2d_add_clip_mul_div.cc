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
 * \file conv2d_add_clip_mul_div.cc
 * \brief tbe conv2d+ add + clip + mul + div  ops fusion pattern
 */

#include "conv2d_add_clip_mul_div.h"
#include <string>
#include <vector>
#include <memory>
#include "graph/utils/tensor_utils.h"
#include "op_log.h"
#include "pattern_fusion_util.h"
#include "graph_optimizer/buffer_fusion/buffer_fusion_pass_registry.h"
#include "conv2d_slice_info_cal_base.h"
#include "tbe_aipp_fusion_rule.h"

namespace fe {
using std::vector;
static const char kPatternConv[] = "conv2d";
static const char kPatternAdd[] = "add";
static const char kPatternAdd1[] = "add1";
static const char kPatternClip[] = "clip";
static const char kPatternMul1[] = "mul";
static const char kPatternDiv[] = "div";
static const char kPatternMuls[] = "muls";
static const char kPatternOtherInput[] = "otherInput";
static const char kPatternOtherInput1[] = "otherInput1";
static const char kPatternOtherInput2[] = "otherInput2";
static const char kPatternOtherInput3[] = "otherInput3";
static const char kPatternOtherInput4[] = "otherInput4";
static const char kPatternOtherInput5[] = "otherInput5";
static const char kPatternOtherInputd[] = "otherInputd";
static const char kPatternOtherOutput[] = "otherOutput";
static const char kPatternOtherOutput1[] = "otherOutput1";

/*
 * @brief:  define conv2d op fusion pattern
 *
 * pattern configuration limit:
 * 1. total min value must be 1 for all head candidated desc.
 * 2. any head candidated desc max value must be 1.
 * 3. output desc can not be itself.
 *        conv2d  -->     add  -->  clip         -->     mul1   -->   div
 *  otheroutput   otherinput1--/    input4,5--/  otherInput /  otherinput2-/
 *             |-------------------------------------------/
 *
 * @return BufferFusionPattern: return all valid patterns.
 */
vector<BufferFusionPattern*> Conv2dAddClipMulDivFusionPass::DefinePatterns()
{
  vector<BufferFusionPattern*> patterns;

  string pattern_name = "ConvAddClipMulDivFusionPass";
  BufferFusionPattern* pattern = new (std::nothrow) BufferFusionPattern(pattern_name);
  FUSION_PASS_CHECK((pattern == nullptr), OP_LOGE(fused_op_type_.c_str(), "new an object failed."), return patterns);
  OP_LOGD(fused_op_type_.c_str(), "Start to define %s pass pattern.", pattern_name.c_str());
 /* define pattern    conv2d  -->                add  -->  clip  -->   mul1   -->    div
 *            otheroutput|           otherinput1--/            otherInput/ otherinput2-/
 *                       |----------------------------------------------/
 */
  pattern->AddOpDesc(kPatternConv, {OP_PATTERN_CONV}, TBE_PATTERN_NUM_DEFAULT, TBE_PATTERN_NUM_DEFAULT)
      .AddOpDesc(kPatternAdd, {OP_PATTERN_ELEMWISE}, TBE_PATTERN_NUM_DEFAULT, TBE_PATTERN_NUM_DEFAULT)
      .AddOpDesc(kPatternClip, {OP_PATTERN_ELEMWISE}, TBE_PATTERN_NUM_DEFAULT, TBE_PATTERN_NUM_DEFAULT)
      .AddOpDesc(kPatternMul1, {OP_PATTERN_ELEMWISE}, TBE_PATTERN_NUM_DEFAULT, TBE_PATTERN_NUM_DEFAULT)
      .AddOpDesc(kPatternDiv, {OP_PATTERN_ELEMWISE}, TBE_PATTERN_NUM_DEFAULT, TBE_PATTERN_NUM_DEFAULT)
      .AddOpDesc(kPatternOtherInput1, {TBE_PATTERN_INPUT_NODE}, TBE_PATTERN_NUM_DEFAULT, TBE_PATTERN_NUM_DEFAULT)
      .AddOpDesc(kPatternOtherInput2, {TBE_PATTERN_INPUT_NODE}, TBE_PATTERN_NUM_DEFAULT, TBE_PATTERN_NUM_DEFAULT)
      .AddOpDesc(kPatternOtherInput3, {TBE_PATTERN_INPUT_NODE}, TBE_PATTERN_NUM_DEFAULT, TBE_PATTERN_NUM_DEFAULT)
      .AddOpDesc(kPatternOtherInput4, {TBE_PATTERN_INPUT_NODE}, TBE_PATTERN_NUM_DEFAULT, TBE_PATTERN_NUM_DEFAULT)
      .SetHead({kPatternConv})
      .SetOutputs(kPatternConv, {kPatternAdd, kPatternMul1}, TBE_OUTPUT_BRANCH_MULTI)
      .SetOutputs(kPatternOtherInput1, {kPatternAdd})
      .SetOutputs(kPatternAdd, {kPatternClip}, TBE_OUTPUT_BRANCH_SINGLE)
      .SetOutputs(kPatternClip, {kPatternMul1}, TBE_OUTPUT_BRANCH_SINGLE)
      .SetOutputs(kPatternMul1, {kPatternDiv}, TBE_OUTPUT_BRANCH_SINGLE)
      .SetOutputs(kPatternOtherInput2, {kPatternDiv})
      .SetOutputs(kPatternOtherInput3, {kPatternClip})
      .SetOutputs(kPatternOtherInput4, {kPatternClip});
  OP_LOGD(fused_op_type_.c_str(), "End to define %s pass pattern.", pattern_name.c_str());
  patterns.push_back(pattern);

  string pattern2_name = "ConvAddClipMulDiv2mulsFusionPass";
  BufferFusionPattern* pattern2 = new (std::nothrow) BufferFusionPattern(pattern2_name);
  FUSION_PASS_CHECK((pattern == nullptr), OP_LOGE(fused_op_type_.c_str(), "new an object failed."), return patterns);
  OP_LOGD(fused_op_type_.c_str(), "Start to define %s pass pattern.", pattern2_name.c_str());
 /* div translate muls on ge fusion if divisor is scalar
 * define pattern    conv2d  -->                add  -->  clip  -->   mul1   -->    muls
 *            otheroutput|           otherinput1--/            otherInput/
 *                       |----------------------------------------------/
 */
  pattern2->AddOpDesc(kPatternConv, {OP_PATTERN_CONV}, TBE_PATTERN_NUM_DEFAULT, TBE_PATTERN_NUM_DEFAULT)
      .AddOpDesc(kPatternAdd, {OP_PATTERN_ELEMWISE}, TBE_PATTERN_NUM_DEFAULT, TBE_PATTERN_NUM_DEFAULT)
      .AddOpDesc(kPatternClip, {OP_PATTERN_ELEMWISE}, TBE_PATTERN_NUM_DEFAULT, TBE_PATTERN_NUM_DEFAULT)
      .AddOpDesc(kPatternMul1, {OP_PATTERN_ELEMWISE}, TBE_PATTERN_NUM_DEFAULT, TBE_PATTERN_NUM_DEFAULT)
      .AddOpDesc(kPatternMuls, {OP_PATTERN_ELEMWISE}, TBE_PATTERN_NUM_DEFAULT, TBE_PATTERN_NUM_DEFAULT)
      .AddOpDesc(kPatternOtherInput1, {TBE_PATTERN_INPUT_NODE}, TBE_PATTERN_NUM_DEFAULT, TBE_PATTERN_NUM_DEFAULT)
      .AddOpDesc(kPatternOtherInput3, {TBE_PATTERN_INPUT_NODE}, TBE_PATTERN_NUM_DEFAULT, TBE_PATTERN_NUM_DEFAULT)
      .AddOpDesc(kPatternOtherInput4, {TBE_PATTERN_INPUT_NODE}, TBE_PATTERN_NUM_DEFAULT, TBE_PATTERN_NUM_DEFAULT)
      .SetHead({kPatternConv})
      .SetOutputs(kPatternConv, {kPatternAdd, kPatternMul1}, TBE_OUTPUT_BRANCH_MULTI)
      .SetOutputs(kPatternOtherInput1, {kPatternAdd})
      .SetOutputs(kPatternAdd, {kPatternClip}, TBE_OUTPUT_BRANCH_SINGLE)
      .SetOutputs(kPatternClip, {kPatternMul1}, TBE_OUTPUT_BRANCH_SINGLE)
      .SetOutputs(kPatternMul1, {kPatternMuls}, TBE_OUTPUT_BRANCH_SINGLE)
      .SetOutputs(kPatternOtherInput3, {kPatternClip})
      .SetOutputs(kPatternOtherInput4, {kPatternClip});
  OP_LOGD(fused_op_type_.c_str(), "End to define %s pass pattern.", pattern2_name.c_str());
  patterns.push_back(pattern2);

  string pattern3_name = "ConvAddClipMulDivBroadcastFusionPass";
  BufferFusionPattern* pattern3 = new (std::nothrow) BufferFusionPattern(pattern3_name);
  FUSION_PASS_CHECK((pattern3 == nullptr), OP_LOGE(fused_op_type_.c_str(), "new an object failed."), return patterns);
  OP_LOGD(fused_op_type_.c_str(), "Start to define %s pass pattern.", pattern3_name.c_str());
 /* define pattern    conv2d  -->                add  -->  clip  -->   mul1   -->    div
 *            otheroutput|           otherinput1--/            otherInput/ otherinput2-/
 *                       |----------------------------------------------/
 */
  pattern3->AddOpDesc(kPatternConv, {OP_PATTERN_CONV}, TBE_PATTERN_NUM_DEFAULT, TBE_PATTERN_NUM_DEFAULT)
      .AddOpDesc(kPatternAdd, {OP_PATTERN_BROAD_CAST}, TBE_PATTERN_NUM_DEFAULT, TBE_PATTERN_NUM_DEFAULT)
      .AddOpDesc(kPatternClip, {OP_PATTERN_BROAD_CAST}, TBE_PATTERN_NUM_DEFAULT, TBE_PATTERN_NUM_DEFAULT)
      .AddOpDesc(kPatternMul1, {OP_PATTERN_BROAD_CAST}, TBE_PATTERN_NUM_DEFAULT, TBE_PATTERN_NUM_DEFAULT)
      .AddOpDesc(kPatternDiv, {OP_PATTERN_BROAD_CAST}, TBE_PATTERN_NUM_DEFAULT, TBE_PATTERN_NUM_DEFAULT)
      .AddOpDesc(kPatternOtherInput1, {TBE_PATTERN_INPUT_NODE}, TBE_PATTERN_NUM_DEFAULT, TBE_PATTERN_NUM_DEFAULT)
      .AddOpDesc(kPatternOtherInput2, {TBE_PATTERN_INPUT_NODE}, TBE_PATTERN_NUM_DEFAULT, TBE_PATTERN_NUM_DEFAULT)
      .AddOpDesc(kPatternOtherInput3, {TBE_PATTERN_INPUT_NODE}, TBE_PATTERN_NUM_DEFAULT, TBE_PATTERN_NUM_DEFAULT)
      .AddOpDesc(kPatternOtherInput4, {TBE_PATTERN_INPUT_NODE}, TBE_PATTERN_NUM_DEFAULT, TBE_PATTERN_NUM_DEFAULT)
      .SetHead({kPatternConv})
      .SetOutputs(kPatternConv, {kPatternAdd, kPatternMul1}, TBE_OUTPUT_BRANCH_MULTI)
      .SetOutputs(kPatternOtherInput1, {kPatternAdd})
      .SetOutputs(kPatternAdd, {kPatternClip}, TBE_OUTPUT_BRANCH_SINGLE)
      .SetOutputs(kPatternClip, {kPatternMul1}, TBE_OUTPUT_BRANCH_SINGLE)
      .SetOutputs(kPatternMul1, {kPatternDiv}, TBE_OUTPUT_BRANCH_SINGLE)
      .SetOutputs(kPatternOtherInput2, {kPatternDiv})
      .SetOutputs(kPatternOtherInput3, {kPatternClip})
      .SetOutputs(kPatternOtherInput4, {kPatternClip});
  OP_LOGD(fused_op_type_.c_str(), "End to define %s pass pattern.", pattern3_name.c_str());
  patterns.push_back(pattern3);

  string pattern4_name = "ConvAddClipMulDiv2mulsBroadcastFusionPass";
  BufferFusionPattern* pattern4 = new (std::nothrow) BufferFusionPattern(pattern4_name);
  FUSION_PASS_CHECK((pattern4 == nullptr), OP_LOGE(fused_op_type_.c_str(), "new an object failed."), return patterns);
  OP_LOGD(fused_op_type_.c_str(), "Start to define %s pass pattern.", pattern4_name.c_str());
 /* div translate muls on ge fusion if divisor is scalar
 * define pattern    conv2d  -->                add  -->  clip  -->   mul1   -->    muls
 *            otheroutput|           otherinput1--/            otherInput/
 *                       |----------------------------------------------/
 */
  pattern4->AddOpDesc(kPatternConv, {OP_PATTERN_CONV}, TBE_PATTERN_NUM_DEFAULT, TBE_PATTERN_NUM_DEFAULT)
      .AddOpDesc(kPatternAdd, {OP_PATTERN_BROAD_CAST}, TBE_PATTERN_NUM_DEFAULT, TBE_PATTERN_NUM_DEFAULT)
      .AddOpDesc(kPatternClip, {OP_PATTERN_BROAD_CAST}, TBE_PATTERN_NUM_DEFAULT, TBE_PATTERN_NUM_DEFAULT)
      .AddOpDesc(kPatternMul1, {OP_PATTERN_BROAD_CAST}, TBE_PATTERN_NUM_DEFAULT, TBE_PATTERN_NUM_DEFAULT)
      .AddOpDesc(kPatternMuls, {OP_PATTERN_BROAD_CAST}, TBE_PATTERN_NUM_DEFAULT, TBE_PATTERN_NUM_DEFAULT)
      .AddOpDesc(kPatternOtherInput1, {TBE_PATTERN_INPUT_NODE}, TBE_PATTERN_NUM_DEFAULT, TBE_PATTERN_NUM_DEFAULT)
      .AddOpDesc(kPatternOtherInput3, {TBE_PATTERN_INPUT_NODE}, TBE_PATTERN_NUM_DEFAULT, TBE_PATTERN_NUM_DEFAULT)
      .AddOpDesc(kPatternOtherInput4, {TBE_PATTERN_INPUT_NODE}, TBE_PATTERN_NUM_DEFAULT, TBE_PATTERN_NUM_DEFAULT)
      .SetHead({kPatternConv})
      .SetOutputs(kPatternConv, {kPatternAdd, kPatternMul1}, TBE_OUTPUT_BRANCH_MULTI)
      .SetOutputs(kPatternOtherInput1, {kPatternAdd})
      .SetOutputs(kPatternAdd, {kPatternClip}, TBE_OUTPUT_BRANCH_SINGLE)
      .SetOutputs(kPatternClip, {kPatternMul1}, TBE_OUTPUT_BRANCH_SINGLE)
      .SetOutputs(kPatternMul1, {kPatternMuls}, TBE_OUTPUT_BRANCH_SINGLE)
      .SetOutputs(kPatternOtherInput3, {kPatternClip})
      .SetOutputs(kPatternOtherInput4, {kPatternClip});
  OP_LOGD(fused_op_type_.c_str(), "End to define %s pass pattern.", pattern4_name.c_str());
  patterns.push_back(pattern4);
  return patterns;
}

Status Conv2dAddClipMulDivFusionPass::GetFusionNodes(const BufferFusionMapping& mapping,
                                                     vector<ge::NodePtr>& fusion_nodes)
{
  OP_LOGD(fused_op_type_.c_str(), "Begin to do Conv2dAddClipMulDiv!");
  fusion_nodes = GetMatchedNodes(mapping);

  FUSION_PASS_CHECK(!BroadcastPatternRule::CheckBroadcastFusionScenario(mapping, fusion_nodes),
                    OP_LOGD(fused_op_type_.c_str(), "The fusion scenario is not applicable"),
                    return NOT_CHANGED);
  OP_LOGD(fused_op_type_.c_str(), "End to do Conv2dAddClipMulDiv.");
  return SUCCESS;
}

Status Conv2dAddClipMulDivFusionPass::CalcFusionOpSliceInfo(vector<ge::NodePtr> &fusion_nodes, OpCalcInfo &op_slice_info)
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

REGISTER_BUFFER_FUSION_PASS("TbeConv2dAddClipMulDivFusionPass", BUILT_IN_AI_CORE_BUFFER_FUSION_PASS,
                            Conv2dAddClipMulDivFusionPass);
}  // namespace fe