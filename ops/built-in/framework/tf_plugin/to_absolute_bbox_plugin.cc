/**
 * Copyright (C)  2020. Huawei Technologies Co., Ltd. All rights reserved.

 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the Apache License Version 2.0. You may not use this file except in compliance with the License.

 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
 * Apache License for more details at
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * @file to_absolute_bbox_plugin.cpp
 *
 * @brief tensorflow plugin for to_absolute_bbox
 *
 * @version 1.0
 *
 */

#include "register/register.h"
#include "tensorflow_fusion_op_parser_util.h"
#include "graph/utils/attr_utils.h"
#include "graph/utils/op_desc_utils.h"
#include "proto/tensorflow/node_def.pb.h"
#include "op_log.h"

using std::vector;
using google::protobuf::Message;
using domi::tensorflow::NodeDef;
using domi::tensorflow::TensorProto;

namespace domi {

static const int kMeanInputSize = 2;
Status ToAbsoluteBBoxParserParams(const std::vector<const google::protobuf::Message *> inside_nodes, ge::Operator &op) {
  OP_LOGI(op.GetName().c_str(), "Enter ToAbsoluteBBox fusion parser.");

  return SUCCESS;
}

REGISTER_CUSTOM_OP("ToAbsoluteBBox")
  .FrameworkType(TENSORFLOW)
  .OriginOpType("ToAbsoluteBBox")
  .FusionParseParamsFn(ToAbsoluteBBoxParserParams)
  .ImplyType(ImplyType::TVM);
}  // namespace domi



