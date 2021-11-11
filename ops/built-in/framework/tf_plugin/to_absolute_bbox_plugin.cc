/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2020. All rights reserved.
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
 * \file to_absolute_bbox_plugin.cpp
 * \brief tensorflow plugin for to_absolute_bbox
 */
#include "register/register.h"
#include "graph/utils/attr_utils.h"
#include "graph/utils/op_desc_utils.h"
#include "proto/tensorflow/node_def.pb.h"

#include "tensorflow_fusion_op_parser_util.h"
#include "op_log.h"

using domi::tensorflow::NodeDef;
using domi::tensorflow::TensorProto;
using google::protobuf::Message;
using std::vector;

namespace domi {
static const int kMeanInputSize = 2;
Status ToAbsoluteBBoxParserParams(const std::vector<const google::protobuf::Message*>& inside_nodes, ge::Operator& op) {
  OP_LOGI(op.GetName().c_str(), "Enter ToAbsoluteBBox fusion parser.");

  return SUCCESS;
}

REGISTER_CUSTOM_OP("ToAbsoluteBBox")
    .FrameworkType(TENSORFLOW)
    .OriginOpType("ToAbsoluteBBox")
    .FusionParseParamsFn(ToAbsoluteBBoxParserParams)
    .ImplyType(ImplyType::TVM);
}  // namespace domi
