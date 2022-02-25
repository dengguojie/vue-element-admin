/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2019. All rights reserved.
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
 * \file group_point_plugin.cc
 * \brief
 */
#include <string>
#include <vector>
#include <map>
#include "register/register.h"
#include "graph/utils/op_desc_utils.h"
#include "proto/tensorflow/node_def.pb.h"
#include "graph/operator.h"
#include "tensorflow_fusion_op_parser_util.h"
#include "graph.h"
#include "op_log.h"
#include "all_ops.h"

namespace domi {
using namespace ge;

Status ParseParamsGroupPoint(const Message *op_src, ge::Operator &op_dest)  {
  const domi::tensorflow::NodeDef *const node_src = ge::PtrToPtr<const ascend_private::protobuf::Message,
                                                                 const domi::tensorflow::NodeDef>(op_src);
  int n = node_src->input_size();
  OP_LOGI(op_dest.GetName().c_str(), "ParseParamsGroupPoint input_size = %d", n);
  // 1.add dynamic input and out
  auto opDesc = ge::OpDescUtils::GetOpDescFromOperator(op_dest);
  if (opDesc == nullptr) {
    OP_LOGE(op_dest.GetName().c_str(), "Failed GetOpDescFromOperator");
    return FAILED;
  }
  opDesc->AddDynamicInputDesc("x", n);
  opDesc->AddDynamicOutputDesc("y", 1);
  // 2.set original_type
  ge::AttrUtils::SetStr(opDesc, "original_type", "GroupPoint");
  // 3.set attr if needed

  return SUCCESS;
}

static Status ParseOpToGraphGroupPoint(const ge::Operator &op, ge::Graph &graph) {
  ge::Operator data_0 = op::Data("points").set_attr_index(0);
  ge::Operator data_1 = op::Data("idx").set_attr_index(1);
  int32_t axis_val = 1;

  TensorDesc tensor1_desc(ge::Shape(), FORMAT_ND, DT_INT32);
  ge::Tensor const_value(tensor1_desc, (uint8_t*)&axis_val, sizeof(axis_val));
  auto const_op = op::Const("const_data").set_attr_value(const_value);
  int batch_dims = 1;
  auto GatherV2 = op::GatherV2().set_input_x(data_0).set_input_indices(data_1)
                  .set_input_axis(const_op).set_attr_batch_dims(batch_dims);
  std::vector<ge::Operator> inputs{data_0, data_1, const_op};
  std::vector<std::pair<ge::Operator, std::vector<size_t>>> output_indexs;
  output_indexs.emplace_back(GatherV2, vector<std::size_t>{0});
  graph.SetInputs(inputs).SetOutputs(output_indexs);

  return SUCCESS;
}

// register GatherPoint op info to GE
REGISTER_CUSTOM_OP("PartitionedCall")
    .FrameworkType(TENSORFLOW)
    .OriginOpType("GroupPoint")
    .ParseParamsFn(ParseParamsGroupPoint)
    .ParseOpToGraphFn(ParseOpToGraphGroupPoint)
    .ImplyType(ImplyType::TVM);
}  // namespace domi
