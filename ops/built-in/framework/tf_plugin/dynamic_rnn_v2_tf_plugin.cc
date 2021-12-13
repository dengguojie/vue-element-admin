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
 * \file dynamic_rnn_v2_tf_plugin.cc
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
constexpr int32_t index_zero = 0;
constexpr int32_t index_one = 1;
constexpr int32_t index_two = 2;
constexpr int32_t index_three = 3;
constexpr int32_t index_four = 4;

Status ParseParamsDynamicRNN(const Message *op_src, ge::Operator &op_dest) {
  // Set original_type
  op_dest.SetAttr("original_type", "DynamicRnnV2");
  return SUCCESS;
}

static Status ParseOpToGraphDynamicRNN(const ge::Operator &op, ge::Graph &graph) {
  ge::Operator data_0 = op::Data("x").set_attr_index(index_zero);
  ge::Operator data_1 = op::Data("w").set_attr_index(index_one);
  ge::Operator data_2 = op::Data("b").set_attr_index(index_two);
  ge::Operator data_3 = op::Data("init_h").set_attr_index(index_three);
  ge::Operator data_4 = op::Data("init_c").set_attr_index(index_four);

  auto rnn = op::DynamicRNN().set_input_x(data_0).set_input_w(data_1).set_input_b(data_2)
                 .set_input_init_h(data_3).set_input_init_c(data_4);

  std::vector<ge::Operator> inputs{data_0, data_1, data_2, data_3, data_4};
  std::vector<std::pair<ge::Operator, std::vector<size_t>>> output_indexs;
  output_indexs.emplace_back(rnn, vector<std::size_t>{0, 1, 2, 3, 4, 5, 6, 7});
  graph.SetInputs(inputs).SetOutputs(output_indexs);
  return SUCCESS;
}

REGISTER_CUSTOM_OP("DynamicRNN")
    .FrameworkType(TENSORFLOW)
    .OriginOpType("DynamicRnnV2")
    .ParseParamsFn(ParseParamsDynamicRNN)
    .ParseOpToGraphFn(ParseOpToGraphDynamicRNN)
    .ImplyType(ImplyType::TVM);
}  // namespace domi
