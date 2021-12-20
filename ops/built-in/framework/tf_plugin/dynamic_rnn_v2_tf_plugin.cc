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
  AutoMappingFn(op_src, op_dest);
  op_dest.SetAttr("is_misplaced", true);
  return SUCCESS;
}

REGISTER_CUSTOM_OP("DynamicRNN")
    .FrameworkType(TENSORFLOW)
    .OriginOpType("DynamicRnnV2")
    .ParseParamsFn(ParseParamsDynamicRNN)
    .ImplyType(ImplyType::TVM);
}  // namespace domi
