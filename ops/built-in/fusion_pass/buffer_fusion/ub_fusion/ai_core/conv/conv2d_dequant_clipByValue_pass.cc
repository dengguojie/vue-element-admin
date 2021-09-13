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
 * \file conv2d_dequant_clipByValue_pass.cpp
 * \brief tbe conv2d + dequant + clipByValue + (quant) ops fusion pattern
 */
#include "conv2d_dequant_clipByValue_pass.h"
#include <string>
#include <vector>
#include <memory>
#include "graph/utils/tensor_utils.h"
#include "op_log.h"
#include "pattern_fusion_util.h"
#include "graph_optimizer/buffer_fusion/buffer_fusion_pass_registry.h"
#include "conv2d_slice_info_cal_base.h"

namespace fe {

static const char PatternConv[] = "conv2d";
static const char PatternDeq[] = "dequant";
static const char PatternElemwise[] = "elemwise";
static const char PatternQuant[] = "quant";
static const char PatternOtherInput1[] = "otherInput1";
static const char PatternOtherInput2[] = "otherInput2";
static const char PatternOtherInput3[] = "otherInput3";

/*
 * @brief:  define conv2d op fusion pattern
 *
 * pattern configuration limit:
 * 1. total min value must be 1 for all head candidated desc.
 * 2. any head candidated desc max value must be 1.
 * 3. output desc can not be itself.
 *
 *    conv2d  -->  dequant  -->  clipByValue
 *     otherinput1--/   otherinput2--/
 *                      otherinput3-/
 *    conv2d  -->  dequant  -->  clipByValue --> quant
 *      otherinput1--/   otherinput2--/
 *                       otherinput3-/
 *    conv2d   -->  clipByValue -->  quant
 *          otherinput1--/
 *          otherinput2-/
 * @return BufferFusionPattern: return all valid patterns.
 */
vector<BufferFusionPattern*> Conv2DDequantClipByValueFusionPass::DefinePatterns() {
  vector<BufferFusionPattern*> patterns;
  string passname = "Conv2DDequantClipByValueQuantFusion";
  BufferFusionPattern *pattern = new (std::nothrow) BufferFusionPattern(passname);
  FUSION_PASS_CHECK((pattern == nullptr), OP_LOGE("new an object failed."), return patterns);
  OP_LOGD("Start to define %s pass pattern.", passName.c_str());
/* define pattern   conv2d  -->  dequant  -->  clipByValue --> quant
 *                    otherinput1--/   otherinput2--/
 *                                     otherinput3-/
*/
  pattern->AddOpDesc(PatternConv, {OP_PATTERN_CONV}, TBE_PATTERN_NUM_DEFAULT, TBE_PATTERN_NUM_DEFAULT)
      .AddOpDesc(PatternDeq, {OP_PATTERN_DEQUANT}, TBE_PATTERN_NUM_DEFAULT, TBE_PATTERN_NUM_DEFAULT)
      .AddOpDesc(PatternElemwise, {OP_PATTERN_ELEMWISE}, TBE_PATTERN_NUM_DEFAULT, TBE_PATTERN_NUM_DEFAULT)
      .AddOpDesc(PatternQuant, {OP_PATTERN_QUANT}, TBE_PATTERN_NUM_DEFAULT, TBE_PATTERN_NUM_DEFAULT)
      .AddOpDesc(PatternOtherInput1, {TBE_PATTERN_INPUT_NODE}, TBE_PATTERN_NUM_DEFAULT, TBE_PATTERN_NUM_DEFAULT)
      .AddOpDesc(PatternOtherInput2, {TBE_PATTERN_INPUT_NODE}, TBE_PATTERN_NUM_DEFAULT, TBE_PATTERN_NUM_DEFAULT)
      .AddOpDesc(PatternOtherInput3, {TBE_PATTERN_INPUT_NODE}, TBE_PATTERN_NUM_DEFAULT, TBE_PATTERN_NUM_DEFAULT)
      .SetHead({PatternConv})
      .SetOutputs(PatternConv, {PatternDeq})
      .SetOutputs(PatternOtherInput1, {PatternDeq})
      .SetOutputs(PatternDeq, {PatternElemwise})
      .SetOutputs(PatternOtherInput2, {PatternElemwise})
      .SetOutputs(PatternOtherInput3, {PatternElemwise})
      .SetOutputs(PatternElemwise, {PatternQuant});
  patterns.push_back(pattern);
  OP_LOGD(fused_op_type_.c_str(), "End to define %s pass pattern.", passname.c_str());

  string passname1 = "Conv2DDequantClipByValueFusion";
  BufferFusionPattern* pattern1 = new (std::nothrow) BufferFusionPattern(passname1);
  FUSION_PASS_CHECK((pattern1 == nullptr), OP_LOGE(fused_op_type_.c_str(), "create new pattern failed."),
                    return patterns);
  OP_LOGD(fused_op_type_.c_str(), "Start to define %s pass pattern.", passname1.c_str());
/* define pattern   conv2d  -->  dequant  -->  clipByValue
 *                    otherinput1--/   otherinput2--/
 *                                     otherinput3-/
*/
  pattern1->AddOpDesc(PatternConv, {OP_PATTERN_CONV}, TBE_PATTERN_NUM_DEFAULT, TBE_PATTERN_NUM_DEFAULT)
      .AddOpDesc(PatternDeq, {OP_PATTERN_DEQUANT}, TBE_PATTERN_NUM_DEFAULT, TBE_PATTERN_NUM_DEFAULT)
      .AddOpDesc(PatternElemwise, {OP_PATTERN_ELEMWISE}, TBE_PATTERN_NUM_DEFAULT, TBE_PATTERN_NUM_DEFAULT)
      .AddOpDesc(PatternOtherInput1, {TBE_PATTERN_INPUT_NODE}, TBE_PATTERN_NUM_DEFAULT, TBE_PATTERN_NUM_DEFAULT)
      .AddOpDesc(PatternOtherInput2, {TBE_PATTERN_INPUT_NODE}, TBE_PATTERN_NUM_DEFAULT, TBE_PATTERN_NUM_DEFAULT)
      .AddOpDesc(PatternOtherInput3, {TBE_PATTERN_INPUT_NODE}, TBE_PATTERN_NUM_DEFAULT, TBE_PATTERN_NUM_DEFAULT)
      .SetHead({PatternConv})
      .SetOutputs(PatternConv, {PatternDeq})
      .SetOutputs(PatternOtherInput1, {PatternDeq})
      .SetOutputs(PatternDeq, {PatternElemwise})
      .SetOutputs(PatternOtherInput2, {PatternElemwise})
      .SetOutputs(PatternOtherInput3, {PatternElemwise});
  patterns.push_back(pattern1);
  OP_LOGD(fused_op_type_.c_str(), "End of defination of %s pass pattern.", passname1.c_str());

  string passname2 = "Conv2DClipByValueQuantFusionPass";
  BufferFusionPattern* pattern2 = new (std::nothrow) BufferFusionPattern(passname2);
  FUSION_PASS_CHECK((pattern2 == nullptr), OP_LOGE(fused_op_type_.c_str(), "create new pattern failed."), return patterns);
  OP_LOGD(fused_op_type_.c_str(), "Start to define %s pass pattern.", passname2.c_str());
/* define pattern   conv2d   -->  clipByValue -->  quant
 *                         otherinput1--/
 *                         otherinput2-/
*/
  pattern2->AddOpDesc(PatternConv, {OP_PATTERN_CONV}, TBE_PATTERN_NUM_DEFAULT, TBE_PATTERN_NUM_DEFAULT)
      .AddOpDesc(PatternElemwise, {OP_PATTERN_ELEMWISE}, TBE_PATTERN_NUM_DEFAULT, TBE_PATTERN_NUM_DEFAULT)
      .AddOpDesc(PatternOtherInput1, {TBE_PATTERN_INPUT_NODE}, TBE_PATTERN_NUM_DEFAULT, TBE_PATTERN_NUM_DEFAULT)
      .AddOpDesc(PatternOtherInput2, {TBE_PATTERN_INPUT_NODE}, TBE_PATTERN_NUM_DEFAULT, TBE_PATTERN_NUM_DEFAULT)
      .AddOpDesc(PatternQuant, {OP_PATTERN_QUANT}, TBE_PATTERN_NUM_DEFAULT, TBE_PATTERN_NUM_DEFAULT)
      .SetHead({PatternConv})
      .SetOutputs(PatternConv, {PatternElemwise})
      .SetOutputs(PatternOtherInput1, {PatternElemwise})
      .SetOutputs(PatternOtherInput2, {PatternElemwise})
      .SetOutputs(PatternElemwise, {PatternQuant});
  patterns.push_back(pattern2);
  OP_LOGD(fused_op_type_.c_str(), "End of defination of %s pass pattern.", passname2.c_str());

  return patterns;
}

Status Conv2DDequantClipByValueFusionPass::CalcFusionOpSliceInfo(vector<ge::NodePtr> &fusion_nodes, OpCalcInfo &op_slice_info){
  OP_LOGD(fused_op_type_.c_str(), "start calc slice info.");
  std::unique_ptr<ConvSliceInfoCalBase> pConvSliceInfoCal = nullptr;
  pConvSliceInfoCal.reset(new (std::nothrow) ConvSliceInfoCalBase());
  CONV_RET_IF_SMART_PTR_IS_NULL(pConvSliceInfoCal);
  Status ret = pConvSliceInfoCal->ConvCalcFusionOpSliceInfo(fusion_nodes, op_slice_info, fused_op_type_);
  FUSION_PASS_CHECK(ret != SUCCESS, OP_LOGW(fused_op_type_.c_str(), "calc fusion op slice info failed."), return FAILED);
  OP_LOGD(fused_op_type_.c_str(), "end calc slice info.");
  return SUCCESS;
}

/*
  * @brief: parse nodes matched in mapping and call DoFusion
  * @param [in] graph: original graph
  * @param [out] mapping: nodes matched by pattern
  * @return bool: fusion status ok or not
 */
Status Conv2DDequantClipByValueFusionPass::GetFusionNodes(const BufferFusionMapping &mapping, vector<ge::NodePtr> &fusionNodes) {
  OP_LOGD(fused_op_type_.c_str(), "Begin of Conv2DDequantClipByValue ub fusion pass!");
  vector<ge::NodePtr> elemNode = GetMatchedNodesByDescName(PatternElemwise, mapping);
  if (!elemNode.empty()) {
    if (elemNode[0]->GetType() != "ClipByValue") {
        OP_LOGI(fused_op_type_.c_str(), "The optype of node[%s] should be ClipByValue, but actually is [%s], no need to do fusion.",
            elemNode[0]->GetName().c_str(), elemNode[0]->GetType().c_str());
        return SUCCESS;
    }
  }

  fusionNodes = GetMatchedNodes(mapping);
  OP_LOGD(fused_op_type_.c_str(), "End of Conv2DDequantClipByValue ub fusion pass!");
  return SUCCESS;
}

REGISTER_BUFFER_FUSION_PASS("Conv2DDequantClipByValueFusionPass",
                            BUILT_IN_AI_CORE_BUFFER_FUSION_PASS,
                            Conv2DDequantClipByValueFusionPass);
}  // namespace fe