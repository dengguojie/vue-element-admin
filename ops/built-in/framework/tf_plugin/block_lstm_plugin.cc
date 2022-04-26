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
 * \file block_lstm_plugin.cpp
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

Status ParseParamsRNN(const Message *op_src, ge::Operator &op_dest) {
  // Set original_type
  op_dest.SetAttr("original_type", "BlockLSTM");
  return SUCCESS;
}

static Status ParseOpToGraphRNN(const ge::Operator &op, ge::Graph &graph) {
  ge::Operator data_0 = op::Data("seq_len_max").set_attr_index(0);
  ge::Operator data_1 = op::Data("x").set_attr_index(1);
  ge::Operator data_2 = op::Data("cs_prev").set_attr_index(2);
  ge::Operator data_3 = op::Data("h_prev").set_attr_index(3);
  ge::Operator data_4 = op::Data("w").set_attr_index(4);
  ge::Operator data_8 = op::Data("b").set_attr_index(8);

  float forget_bias = 0.0;
  if (op.GetAttr("forget_bias", forget_bias) != ge::GRAPH_SUCCESS) {
    OP_LOGE(TbeGetName(op).c_str(), "get attr forget_bias failed.");
    return FAILED;
  }

  float cell_clip = 3.0;
  if (op.GetAttr("cell_clip", cell_clip) != ge::GRAPH_SUCCESS) {
    OP_LOGE(TbeGetName(op).c_str(), "get attr cell_clip failed.");
    return FAILED;
  }

  bool use_peephole = false;
  if (op.GetAttr("use_peephole", use_peephole) != ge::GRAPH_SUCCESS) {
    OP_LOGE(TbeGetName(op).c_str(), "get attr use_peephole failed.");
    return FAILED;
  }

  auto cast = op::Cast().set_input_x(data_0).set_attr_dst_type(3);
  auto rnn = op::DynamicRNN().set_input_x(data_1).set_input_w(data_4).set_input_b(data_8).set_input_seq_length(cast)
                 .set_input_init_h(data_3).set_input_init_c(data_2).set_attr_forget_bias(forget_bias)
                 .set_attr_cell_clip(cell_clip).set_attr_use_peephole(use_peephole);

  std::vector<ge::Operator> inputs{data_1, data_4, data_8, data_0, data_3, data_2};
  std::vector<std::pair<ge::Operator, std::vector<size_t>>> output_indexs;
  output_indexs.emplace_back(rnn, vector<std::size_t>{3, 2, 5, 6, 4, 7, 1});
  graph.SetInputs(inputs).SetOutputs(output_indexs);
  return SUCCESS;
}

// register BlockLSTM op info to GE
REGISTER_CUSTOM_OP("DynamicRNN")
    .FrameworkType(TENSORFLOW)
    .OriginOpType("BlockLSTM")
    .ParseParamsFn(ParseParamsRNN)
    .ParseOpToGraphFn(ParseOpToGraphRNN)
    .ImplyType(ImplyType::TVM);
}  // namespace domi
