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
 * \file conv2d_add_relu6_mul_mul.cc
 * \brief tbe conv2d+ add + relu6 + mul + mul  ops fusion pattern
 */

#include "conv2d_add_relu6_mul_mul.h"
#include <string>
#include <vector>
#include <memory>
#include "graph/utils/tensor_utils.h"
#include "op_log.h"
#include "pattern_fusion_util.h"
#include "graph_optimizer/buffer_fusion/buffer_fusion_pass_registry.h"
#include "conv2d_slice_info_cal_base.h"

namespace fe {
using std::vector;
static const char kPatternConv[] = "conv2d";
static const char kPatternDWConv[] = "DepthwiseConvolution";
static const char kPatternAdd[] = "add";
static const char kPatternRelu6[] = "relu6";
static const char kPatternMul1[] = "mul1";
static const char kPatternMul2[] = "mul2";
static const char kPatternDeq[] = "dequant";
static const char kPatternQuant[] = "quant";
static const char kPatternOtherInput[] = "otherInput";
static const char kPatternOtherInput1[] = "otherInput1";
static const char kPatternOtherInput2[] = "otherInput2";
static const char kPatternOtherInput3[] = "otherInput3";
static const char kPatternOtherInputd[] = "otherInputd";
static const char kPatternOtherOutput[] = "otherOutput";
static const char kPatternOtherOutputd[] = "otherOutputd";
static const char kPatternOtherOutput1[] = "otherOutput1";

/*
 * @brief:  define conv2d op fusion pattern
 *
 * pattern configuration limit:
 * 1. total min value must be 1 for all head candidated desc.
 * 2. any head candidated desc max value must be 1.
 * 3. output desc can not be itself.
 *        conv2d(depthwise)   -->     add  -->  relu6  -->   mul1   -->   mul2
 *  otheroutput|           otherinput1--/         otherInput /  otherinput2-/
 *             |--------------------------------------------/

 *        otherinput1--\
 *                      \
 *  Depthwiseconv-->  deuqnt   -->    add   -->  relu6  -->   mul1    -->    mul2  -->  quant
 *        otherOutputd | otherinput2--/           otherInputd /   otherinput3-/ \-->otheroutput1
 *                     |------------------------------------/
 *
 * @return BufferFusionPattern: return all valid patterns.
 */
vector<BufferFusionPattern*> Conv2dAddRelu6MulMulFusionPass::DefinePatterns()
{
  vector<BufferFusionPattern*> patterns;

  string pattern_name = "ConvAddRelu6MulMulFusionPass";
  BufferFusionPattern* pattern = new (std::nothrow) BufferFusionPattern(pattern_name);
  FUSION_PASS_CHECK((pattern == nullptr), OP_LOGE(fused_op_type_.c_str(), "new an object failed."), return patterns);
  OP_LOGD(fused_op_type_.c_str(), "Start to define %s pass pattern.", pattern_name.c_str());
 /* define pattern    conv2d(depthwise)   -->     add  -->  relu6  -->   mul1   -->    mul2
 *            otheroutput|           otherinput1--/            otherInput/ otherinput2-/
 *                       |----------------------------------------------/
 */
  pattern->AddOpDesc(kPatternConv, {OP_PATTERN_CONV, OP_PATTERN_DEPTHWISE_CONV}, TBE_PATTERN_NUM_DEFAULT, TBE_PATTERN_NUM_DEFAULT)
      .AddOpDesc(kPatternAdd, {OP_PATTERN_ELEMWISE}, TBE_PATTERN_NUM_DEFAULT, TBE_PATTERN_NUM_DEFAULT)
      .AddOpDesc(kPatternRelu6, {OP_PATTERN_ELEMWISE}, TBE_PATTERN_NUM_DEFAULT, TBE_PATTERN_NUM_DEFAULT)
      .AddOpDesc(kPatternMul1, {OP_PATTERN_ELEMWISE}, TBE_PATTERN_NUM_DEFAULT, TBE_PATTERN_NUM_DEFAULT)
      .AddOpDesc(kPatternMul2, {OP_PATTERN_ELEMWISE}, TBE_PATTERN_NUM_DEFAULT, TBE_PATTERN_NUM_DEFAULT)
      .AddOpDesc(kPatternOtherInput1, {TBE_PATTERN_INPUT_NODE}, TBE_PATTERN_NUM_DEFAULT, TBE_PATTERN_NUM_DEFAULT)
      .AddOpDesc(kPatternOtherInput2, {TBE_PATTERN_INPUT_NODE}, TBE_PATTERN_NUM_DEFAULT, TBE_PATTERN_NUM_DEFAULT)
      .AddOpDesc(kPatternOtherInput, {TBE_PATTERN_INPUT_NODE}, TBE_PATTERN_NUM_DEFAULT, TBE_PATTERN_NUM_DEFAULT)
      .AddOpDesc(kPatternOtherOutput, {TBE_PATTERN_OUTPUT_NODE}, TBE_PATTERN_NUM_DEFAULT, TBE_PATTERN_NUM_DEFAULT)
      .SetHead({kPatternConv})
      .SetOutputs(kPatternConv, {kPatternAdd, kPatternOtherOutput}, TBE_OUTPUT_BRANCH_MULTI)
      .SetOutputs(kPatternOtherInput1, {kPatternAdd})
      .SetOutputs(kPatternAdd, {kPatternRelu6}, TBE_OUTPUT_BRANCH_SINGLE)
      .SetOutputs(kPatternRelu6, {kPatternMul1}, TBE_OUTPUT_BRANCH_SINGLE)
      .SetOutputs(kPatternMul1, {kPatternMul2}, TBE_OUTPUT_BRANCH_SINGLE)
      .SetOutputs(kPatternOtherInput, {kPatternMul1})
      .SetOutputs(kPatternOtherInput2, {kPatternMul2});
  patterns.push_back(pattern);
  OP_LOGD(fused_op_type_.c_str(), "End to define %s pass pattern.", pattern_name.c_str());

  string pattern_name1 = "DepthwiseconvDequantAddRelu6MulMulQuantFusionPass";
  BufferFusionPattern* pattern1 = new (std::nothrow) BufferFusionPattern(pattern_name1);
  FUSION_PASS_CHECK((pattern1 == nullptr), OP_LOGE(fused_op_type_.c_str(), "new an object failed."), return patterns);
  OP_LOGD(fused_op_type_.c_str(), "Start to define %s pass pattern.", pattern_name1.c_str());

/* define pattern
 *       otherinput1--\
 *                     \
 *  Depthwiseconv-->  deuqnt   -->    add   -->  relu6  -->   mul1    -->    mul2  -->  quant
 *        otherOutputd | otherinput2--/          otherInputd /   otherinput3-/  \-->otheroutput1
 *                     |------------------------------------/
 */
  pattern1->AddOpDesc(kPatternDWConv, {OP_PATTERN_CONV}, TBE_PATTERN_NUM_DEFAULT, TBE_PATTERN_NUM_DEFAULT)
      .AddOpDesc(kPatternDeq, {OP_PATTERN_DEQUANT}, TBE_PATTERN_NUM_DEFAULT, TBE_PATTERN_NUM_DEFAULT)
      .AddOpDesc(kPatternQuant, {OP_PATTERN_QUANT}, TBE_PATTERN_NUM_NONE, TBE_PATTERN_NUM_DEFAULT)
      .AddOpDesc(kPatternAdd, {OP_PATTERN_ELEMWISE}, TBE_PATTERN_NUM_DEFAULT, TBE_PATTERN_NUM_DEFAULT)
      .AddOpDesc(kPatternRelu6, {OP_PATTERN_ELEMWISE}, TBE_PATTERN_NUM_DEFAULT, TBE_PATTERN_NUM_DEFAULT)
      .AddOpDesc(kPatternMul1, {OP_PATTERN_ELEMWISE}, TBE_PATTERN_NUM_DEFAULT, TBE_PATTERN_NUM_DEFAULT)
      .AddOpDesc(kPatternMul2, {OP_PATTERN_ELEMWISE}, TBE_PATTERN_NUM_DEFAULT, TBE_PATTERN_NUM_DEFAULT)
      .AddOpDesc(kPatternOtherInputd, {TBE_PATTERN_INPUT_NODE}, TBE_PATTERN_NUM_DEFAULT, TBE_PATTERN_NUM_DEFAULT)
      .AddOpDesc(kPatternOtherInput1, {TBE_PATTERN_INPUT_NODE}, TBE_PATTERN_NUM_DEFAULT, TBE_PATTERN_NUM_DEFAULT)
      .AddOpDesc(kPatternOtherInput2, {TBE_PATTERN_INPUT_NODE}, TBE_PATTERN_NUM_DEFAULT, TBE_PATTERN_NUM_DEFAULT)
      .AddOpDesc(kPatternOtherInput3, {TBE_PATTERN_INPUT_NODE}, TBE_PATTERN_NUM_DEFAULT, TBE_PATTERN_NUM_DEFAULT)
      .AddOpDesc(kPatternOtherOutputd, {TBE_PATTERN_OUTPUT_NODE}, TBE_PATTERN_NUM_DEFAULT, TBE_PATTERN_NUM_DEFAULT)
      .SetHead({kPatternDWConv})
      .SetOutputs(kPatternDWConv, {kPatternDeq})
      .SetOutputs(kPatternOtherInput1, {kPatternDeq})
      .SetOutputs(kPatternDeq, {kPatternAdd,  kPatternOtherOutputd}, TBE_OUTPUT_BRANCH_MULTI)
      .SetOutputs(kPatternOtherInput2, {kPatternAdd})
      .SetOutputs(kPatternAdd, {kPatternRelu6}, TBE_OUTPUT_BRANCH_SINGLE)
      .SetOutputs(kPatternRelu6, {kPatternMul1}, TBE_OUTPUT_BRANCH_SINGLE)
      .SetOutputs(kPatternOtherInputd, {kPatternMul1})
      .SetOutputs(kPatternMul1, {kPatternMul2}, TBE_OUTPUT_BRANCH_SINGLE)
      .SetOutputs(kPatternOtherInput3, {kPatternMul2})
      .SetOutputs(kPatternMul2, {kPatternQuant});
  patterns.push_back(pattern1);
  OP_LOGD(fused_op_type_.c_str(), "End to define %s pass pattern.", pattern_name1.c_str());

  return patterns;
}

Status Conv2dAddRelu6MulMulFusionPass::GetFusionNodes(const BufferFusionMapping &mapping,
                                                      vector<ge::NodePtr> &fusion_nodes)
{
  OP_LOGD(fused_op_type_.c_str(), "Begin to do Conv2dAddRelu6MulMulFusionPass.");
  auto kPatternAdd_nodes = GetMatchedNodesByDescName(kPatternAdd, mapping);
  auto kPatternRelu6_nodes = GetMatchedNodesByDescName(kPatternRelu6, mapping);
  auto kPatternMul1_nodes = GetMatchedNodesByDescName(kPatternMul1, mapping);
  auto kPatternMul2_nodes = GetMatchedNodesByDescName(kPatternMul2, mapping);
  auto OtherOutput_nodes = GetMatchedNodesByDescName(kPatternOtherOutput, mapping);
  auto OtherInput_nodes = GetMatchedNodesByDescName(kPatternOtherInput, mapping);
  auto OtherOutputd_nodes = GetMatchedNodesByDescName(kPatternOtherOutputd, mapping);
  auto OtherInputd_nodes = GetMatchedNodesByDescName(kPatternOtherInputd, mapping);
  FUSION_PASS_CHECK(kPatternAdd_nodes.empty() || kPatternAdd_nodes[0]->GetType() != "Add",
                    OP_LOGD(fused_op_type_.c_str(), "Add not support ub fusion"),
                    return SUCCESS);
  FUSION_PASS_CHECK(kPatternRelu6_nodes.empty() || kPatternRelu6_nodes[0]->GetType() != "Relu6",
                    OP_LOGD(fused_op_type_.c_str(), "Relu6 not support ub fusion"),
                    return SUCCESS);
  FUSION_PASS_CHECK(kPatternMul1_nodes.empty() || kPatternMul1_nodes[0]->GetType() != "Mul",
                    OP_LOGD(fused_op_type_.c_str(), "Mul1 not support ub fusion"),
                    return SUCCESS);
  FUSION_PASS_CHECK(kPatternMul2_nodes.empty() || kPatternMul2_nodes[0]->GetType() != "Mul",
                    OP_LOGD(fused_op_type_.c_str(), "Mul2 not support ub fusion"),
                    return SUCCESS);
  if (!OtherOutput_nodes.empty()) {
    if ((OtherOutput_nodes[0]->GetType() != "Mul")) {
      OP_LOGI(fused_op_type_.c_str(),
              "The optype of node[%s] should be Mul, but actually is [%s], no need to do fusion.",
              OtherOutput_nodes[0]->GetName().c_str(), OtherOutput_nodes[0]->GetType().c_str());
      return SUCCESS;
    }
  }
  if (!OtherInput_nodes.empty()) {
    if ((OtherInput_nodes[0]->GetType() != "Depthwiseconv") || (OtherInput_nodes[0]->GetType() != "Conv2d")) {
      OP_LOGI(fused_op_type_.c_str(),
              "The optype of node[%s] should be conv or depthwiseconv, but actually is [%s], no need to do fusion.",
              OtherInput_nodes[0]->GetName().c_str(), OtherInput_nodes[0]->GetType().c_str());
      return SUCCESS;
    }
  }
  if (!OtherOutputd_nodes.empty()) {
    if ((OtherOutputd_nodes[0]->GetType() != "Mul")) {
      OP_LOGI(fused_op_type_.c_str(),
              "The optype of node[%s] should be Mul, but actually is [%s], no need to do fusion.",
              OtherOutputd_nodes[0]->GetName().c_str(), OtherOutputd_nodes[0]->GetType().c_str());
      return SUCCESS;
    }
  }
  if (!OtherInputd_nodes.empty()) {
    if ((OtherInputd_nodes[0]->GetType() != "Dequant")) {
      OP_LOGI(fused_op_type_.c_str(),
              "The optype of node[%s] should be Dequant, but actually is [%s], no need to do fusion.",
              OtherInputd_nodes[0]->GetName().c_str(), OtherInputd_nodes[0]->GetType().c_str());
      return SUCCESS;
    }
  }
  fusion_nodes = GetMatchedNodes(mapping);
  OP_LOGD(fused_op_type_.c_str(), "End of Conv2dAddRelu6MulMul ub fusion pass.");
  return SUCCESS;
}

Status Conv2dAddRelu6MulMulFusionPass::CalcFusionOpSliceInfo(vector<ge::NodePtr> &fusion_nodes, OpCalcInfo &op_slice_info)
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

REGISTER_BUFFER_FUSION_PASS("TbeConv2dAddRelu6MulMulFusionPass", BUILT_IN_AI_CORE_BUFFER_FUSION_PASS,
                            Conv2dAddRelu6MulMulFusionPass);
}  // namespace fe