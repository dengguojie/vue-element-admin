/**
 * Copyright 2021 Huawei Technologies Co., Ltd
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
 * \file mul_fusion_optimizer_pass.cc
 * \brief Define MUL fusion optimize pass
 */
#include "mul_fusion_optimizer_pass.h"

#include <iostream>
#include <map>
#include <string>
#include <vector>
#include <algorithm>

#include "op_log.h"
#include "error_util.h"
#include "pattern_fusion_util.h"
#include "graph/debug/ge_attr_define.h"
#include "graph/utils/attr_utils.h"
#include "graph/utils/graph_utils.h"
#include "graph/utils/node_utils.h"
#include "graph/utils/op_desc_utils.h"
#include "graph/utils/tensor_utils.h"
#include "graph/utils/type_utils.h"
#include "graph_optimizer/graph_fusion/fusion_pass_manager/fusion_pass_registry.h"
#include "graph/operator_reg.h"

using namespace ge;
namespace fe {
const string kPatternMulFusOtimizer = "MUL";
const string kMulType = "Mul";
const string kFusedOptype = "mul_fusion_optimizer";
const std::unordered_set<Format> kOriFormat = {FORMAT_NCHW, FORMAT_NHWC, FORMAT_HWCN, FORMAT_CHWN};

vector<FusionPattern*> MulFusionOptimizeFusionPass::DefinePatterns() {
  vector<FusionPattern*> patterns;
  FusionPattern* pattern = new (std::nothrow) FusionPattern("MulFusionOptimizeFusionPass");
  OP_LOGD(kFusedOptype.c_str(), "Enter MulFusionOptimizeFusionPasss::DefinePatterns.");
  FUSION_PASS_CHECK(pattern == nullptr, VECTOR_FUSION_INNER_ERR_REPORT(kFusedOptype.c_str(), "new an object failed."),
                    return patterns);

  pattern->AddOpDesc(kPatternMulFusOtimizer, {kMulType})
      .SetOutput(kPatternMulFusOtimizer);
  patterns.push_back(pattern);

  return patterns;
}
Status SetNewFormat(ge::GeTensorDescPtr mul_in0, ge::GeTensorDescPtr mul_in1, ge::GeTensorDescPtr mul_out0) {
  auto op_input0_desc_format = mul_in0->GetFormat();
  auto op_input1_desc_format = mul_in1->GetFormat();
  auto op_output0_desc_format = mul_out0->GetFormat();
  auto op_input0_desc_ori_format = mul_in0->GetOriginFormat();
  auto op_input1_desc_ori_format = mul_in1->GetOriginFormat();
  auto op_output0_desc_ori_format = mul_out0->GetOriginFormat();

  OP_LOGD(kFusedOptype.c_str(), "Mul before modify format:input0:%s, input1:%s, output0:%s",
          ge::TypeUtils::FormatToSerialString(op_input0_desc_format).c_str(),
          ge::TypeUtils::FormatToSerialString(op_input1_desc_format).c_str(),
          ge::TypeUtils::FormatToSerialString(op_output0_desc_format).c_str());

  bool change = false;
  if (op_input0_desc_ori_format != op_input0_desc_format) {
    mul_in0->SetFormat(op_input0_desc_ori_format);
    mul_in0->SetShape(mul_in0->GetOriginShape());
    change = true;
  }
  if (op_input1_desc_ori_format != op_input1_desc_format) {
    mul_in1->SetFormat(op_input1_desc_ori_format);
    mul_in1->SetShape(mul_in1->GetOriginShape());
    change = true;
  }
  if (op_output0_desc_ori_format != op_output0_desc_format) {
    mul_out0->SetFormat(op_output0_desc_ori_format);
    mul_out0->SetShape(mul_out0->GetOriginShape());
    change = true;
  }
  if (!change) {
    OP_LOGD(kFusedOptype.c_str(), "all mul tensor format equal ori format, no need to set new format");
    return NOT_CHANGED;
  }
  op_input0_desc_format = mul_in0->GetFormat();
  op_input1_desc_format = mul_in1->GetFormat();
  op_output0_desc_format = mul_out0->GetFormat();

  OP_LOGI(kFusedOptype.c_str(), "Mul after modify format:input0:%s, input1:%s, output0:%s",
          ge::TypeUtils::FormatToSerialString(op_input0_desc_format).c_str(),
          ge::TypeUtils::FormatToSerialString(op_input1_desc_format).c_str(),
          ge::TypeUtils::FormatToSerialString(op_output0_desc_format).c_str());
  return SUCCESS;
}

Status JudgeFormatOK(const ge::GeTensorDescPtr &tensor0, const ge::GeTensorDescPtr &tensor1,
                     const ge::OpDescPtr &op_desc, int &diff_cnt) {
  auto tensor0_format = tensor0->GetFormat();
  auto tensor1_format = tensor1->GetFormat();

  OP_LOGD(kFusedOptype.c_str(), "MulFusionOptimizeFusionPass op %s 's format, tensor0:%s, tensor1:%s",
          op_desc->GetName().c_str(),
          ge::TypeUtils::FormatToSerialString(tensor0_format).c_str(),
          ge::TypeUtils::FormatToSerialString(tensor1_format).c_str());

  if ((kOriFormat.count(tensor0_format) > 0 && tensor1_format == FORMAT_ND) ||
      (kOriFormat.count(tensor1_format) > 0 && tensor0_format == FORMAT_ND)) {
    OP_LOGD(kFusedOptype.c_str(), "MulFusionOptimizeFusionPass op %s's edge, one is ori format, and another is ND");
  } else if (tensor0_format != tensor1_format) {
    diff_cnt++;
  }
  return SUCCESS;
}

/* The input size must be 2, output size must be 1, dtype must be same, and the tensor must be valid
 *   input0        intput1
 *          \     /
              MUL
               |
               |
            outout0
 */
Status MulFusionOptimizeFusionPass::CheckParameterAndSet(ge::NodePtr& in_node_ptr) {
  auto input_nodes = in_node_ptr->GetInDataNodes();
  auto output_nodes = in_node_ptr->GetOutDataNodes();
  auto input_num = input_nodes.size();
  auto output_num = output_nodes.size();
  FUSION_PASS_CHECK((input_num != 2 || output_num != 1),
                    OP_LOGD(kFusedOptype.c_str(), "the input size (%lu) != 2 or "
                            "out size (%lu) != 1", input_num, output_num), return PARAM_INVALID);
  auto node_input0 = input_nodes.at(0);
  auto node_input1 = input_nodes.at(1);
  auto node_output = output_nodes.at(0);

  auto op_desc = in_node_ptr->GetOpDesc();
  auto input_tensor0 = op_desc->MutableInputDesc(0);
  auto input_tensor1 = op_desc->MutableInputDesc(1);
  auto output_tensor = op_desc->MutableOutputDesc(0);
  FUSION_PASS_CHECK((!input_tensor0 || !input_tensor1 || !output_tensor),
                    OP_LOGI(kFusedOptype.c_str(), "The tensor input or output of %s is null!",
                            in_node_ptr->GetName().c_str()), return NOT_CHANGED);

  auto op_input0_desc_dtype = input_tensor0->GetDataType();
  FUSION_PASS_CHECK((op_input0_desc_dtype != DT_FLOAT16 && op_input0_desc_dtype != DT_FLOAT),
                    OP_LOGI(kFusedOptype.c_str(), "The dtype of input0  %s invalid", in_node_ptr->GetName().c_str()),
                    return NOT_CHANGED);

  auto input0_peer_node_desc = node_input0->GetOpDesc();
  auto input1_peer_node_desc = node_input1->GetOpDesc();
  auto output0_peer_node_desc = node_output->GetOpDesc();
  auto input0_peer_output0_tensor = input0_peer_node_desc->MutableOutputDesc(0);
  auto input1_peer_output0_tensor = input1_peer_node_desc->MutableOutputDesc(0);
  uint32_t output0_peer_input0_tensor_idx = in_node_ptr->GetOutDataAnchor(0)->GetPeerInDataAnchors().at(0)->GetIdx();
  auto output0_peer_input0_tensor = output0_peer_node_desc->MutableInputDesc(output0_peer_input0_tensor_idx);
  FUSION_PASS_CHECK((!input0_peer_output0_tensor || !input1_peer_output0_tensor || !output0_peer_input0_tensor),
                    OP_LOGI(kFusedOptype.c_str(), "The tensor input or output of %s is null!",
                            in_node_ptr->GetName().c_str()), return NOT_CHANGED);
  int diff_cnt = 0;
  JudgeFormatOK(input_tensor0, input0_peer_output0_tensor, op_desc, diff_cnt);
  JudgeFormatOK(input_tensor1, input1_peer_output0_tensor, op_desc, diff_cnt);
  JudgeFormatOK(output_tensor, output0_peer_input0_tensor, op_desc, diff_cnt);
  FUSION_PASS_CHECK((diff_cnt != 2), OP_LOGI(kFusedOptype.c_str(), "diff_cnt:%d != 2", diff_cnt), return NOT_CHANGED);

  FUSION_PASS_CHECK(SetNewFormat(input_tensor0, input_tensor1, output_tensor) != SUCCESS,
                    OP_LOGW(kFusedOptype.c_str(), "SetNewFormat failed."), return NOT_CHANGED);

  return SUCCESS;
}

Status MulFusionOptimizeFusionPass::Fusion(ge::ComputeGraph& graph, Mapping& mapping, vector<ge::NodePtr>& new_nodes) {
  OP_LOGD(kFusedOptype.c_str(), "Enter MulFusionOptimizeFusionPass");
  ge::NodePtr in_node_ptr = GetNodeFromMapping(kPatternMulFusOtimizer, mapping);
  FUSION_PASS_CHECK(in_node_ptr == nullptr, VECTOR_FUSION_INNER_ERR_REPORT(kFusedOptype.c_str(), "Node MUL is null, "
                                                                      "fusion failed."), return PARAM_INVALID);
  OP_LOGD(kFusedOptype.c_str(), "Check MulFusionOptimizeFusionPass");
  FUSION_PASS_CHECK(CheckParameterAndSet(in_node_ptr) != SUCCESS, OP_LOGW(kFusedOptype.c_str(), "Check MUL param failed."),
                    return NOT_CHANGED);

  OP_LOGI(kFusedOptype.c_str(), "End MulFusionOptimizeFusionPass");
  return SUCCESS;
}

}  // namespace fe
