
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
 * \file caffe_batch_matmul_plugin.cpp
 * \brief
 */
#include "proto/caffe/caffe.pb.h"
#include "register/register.h"
#include "../../op_proto/util/error_util.h"
#include "common/util/error_manager/error_manager.h"
#include "op_log.h"

namespace domi {

Status ParseParamBatchMatMul(const Message* op_src, ge::Operator& op_dest) {
  // set the default adj_x1 and adj_x2 value for BatchMatMul,

  auto layer = dynamic_cast<const caffe::LayerParameter*>(op_src);
  // Ckeck operator parameter's validity
  if (nullptr == layer) {
    CUBE_INNER_ERR_REPORT_PLUGIN(op_dest.GetName().c_str(), "convert src op failed.");
    return FAILED;
  }
  // get layer
  const caffe::BatchMatMulParameter& param = layer->batch_matmul_param();
  if (param.has_adj_x1()) {
    op_dest.SetAttr("adj_x1", static_cast<bool>(param.adj_x1()));
  } else {
    op_dest.SetAttr("adj_x1", false);
  }

  if (param.has_adj_x2()) {
    op_dest.SetAttr("adj_x2", static_cast<bool>(param.adj_x2()));
  } else {
    op_dest.SetAttr("adj_x2", false);
  }

  return SUCCESS;
}

// register BatchMatMul op info to GE
REGISTER_CUSTOM_OP("BatchMatMul")
    .FrameworkType(CAFFE)
    .OriginOpType("BatchedMatMul")
    .ParseParamsFn(ParseParamBatchMatMul)
    .ImplyType(ImplyType::TVM);
}  // namespace domi
